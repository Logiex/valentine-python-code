import asyncio
import datetime
from enum import Enum
import random
from typing import Dict, List, Literal, Optional, Tuple
from beanie import Document, Link, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import IndexModel
import strawberry
from utils.context import Info
from transport.id import MongoID
from typing import Set, FrozenSet
from .interests import Interests as interest_list
from beanie.operators import In
from scipy.optimize import linear_sum_assignment

@strawberry.enum
class Gender(Enum):
    M = "M"
    F = "F"

class Interest(BaseModel):
    name : str
    score : int

@strawberry.experimental.pydantic.input(model=Interest)
class GQLInterestInput:
    name : str
    score : int

@strawberry.experimental.pydantic.type(model=Interest)
class GQLInterestOut:
    name : str
    score : int
   
class ValentineProfile(Document):
    name : str
    gender : Gender
    wants : Gender
    interests : List[Interest] = Field(default=[])
    clerk_id : str
    email : str
    discord : str
    instagram : str
    friend_only : bool = Field(default=False)

    def __hash__(self):  # make hashable BaseModel subclass
        return hash((type(self),) + tuple(self.dict(include={"name", "gender", "wants", "clerk_id"})))


@strawberry.experimental.pydantic.input(model=ValentineProfile)
class ValentineProfileInput:
    name : str
    gender : Gender
    wants : Gender
    interests : List[GQLInterestInput] = Field(default=[])
    email : str
    discord : str
    instagram : str
    friend_only : bool
        
@strawberry.experimental.pydantic.type(model=ValentineProfile)
class ValentineProfileOut:
    name : str
    gender : Gender
    wants : Gender
    interests : List[GQLInterestOut] = Field(default=[])
    email : str
    id : MongoID 
    discord : str = Field(default="")
    instagram : str = Field(default="")
    
class Participant(BaseModel):
    name : str
    sex : Literal["M" , "F"]
    wants : Literal["M" , "F"]
    friend_only : bool = Field(default=True)
    id : PydanticObjectId

    def __hash__(self):  # make hashable BaseModel subclass
        return hash((type(self),) + tuple(self.dict(include={"name", "sex", "wants", "id"})))

    def want_eachother(self, other_participant:"Participant"):
        return self.sex == other_participant.wants and self.wants == other_participant.sex

    def is_straight(self):
        return self.sex == "F" and self.wants == "M" or self.sex == "M" and self.wants == "F" 

    def is_lesbian(self):
        return self.sex == "F" and self.wants == "F"
        
    def is_gay(self):
        return self.sex == "M" and self.wants == "M"
    
    def calculate_match(self, other : "Participant", interests : dict[str, dict["Participant", int]], randomize=False, previous_matchings:Set[FrozenSet["Participant"]]=set() ):
        tally = 0
        fs = frozenset([self, other])
        if fs in previous_matchings:
            return -1000
        if randomize:
            return random.randint(0, 25) * len(interests)
        for interest in interests.keys():
            mine = interests[interest].get(self, 0)
            theirs = interests[interest].get(other, 0)
            tally += mine * theirs
        return tally
    
    @staticmethod
    def from_valentine_profile(profile : ValentineProfile) -> "Participant":
        return Participant(name=profile.name, sex=profile.gender.name, wants=profile.wants.name, id=profile.id, friend_only=profile.friend_only)

MatchType = Literal["Bro Matching" , "Romantic Matching"]

class Matching(Document):
    participants : Optional[List[Participant]]
    score : int
    time : datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    round : int = Field(default=0)
    match_type : MatchType = Field(default="Romantic Matching")
    valentine_participants : List[Link[ValentineProfile]] = Field(default=[])

    def __hash__(self):  # make hashable BaseModel subclass
        return hash((type(self),) + tuple(self.dict(include={"particants", "score", "time", "round"})))

    class Settings:
        indexes = [IndexModel("valentine_participants"), "round"]

@strawberry.experimental.pydantic.type(model=Matching)
class MatchOut:
    score : strawberry.auto
    round : strawberry.auto
    match_type : str
    time :strawberry.auto
    valentines : List[ValentineProfileOut]

    @staticmethod
    def from_pydantic(instance : Matching) -> "MatchOut":
        intermediate = [ValentineProfileOut.from_pydantic(val_prof) for val_prof in instance.valentine_participants]

        res = instance.dict(include={Matching.score, Matching.round, Matching.match_type, Matching.time} )
        res["valentines"] = intermediate
        return MatchOut(**res)
    ...

class Intermediate(BaseModel):
    p1 : List[Participant] = Field(default=[])
    p2 : List[Participant] = Field(default=[])
    double_arr : List[List[int]]

    def from_graph(graph: dict[Participant, dict[Participant, int]]):
        p1 = [x for x in graph.keys()]
        st :Set[Participant] = set() 
        for x in graph.values():
            for y in x.keys():
                if y not in st:
                    st.add(y)
        p2 = list(st)
        arr: List[List[Participant]] = []
        for x in p1:
            sub:List[Participant] = []
            for y in p2:
                sub.append(graph[x][y])

            arr.append(sub)    
        
        return Intermediate(p1=p1, p2=p2, double_arr=arr)

    def parse_matching(self, rows : List[int], cols:List[int]) -> List[Tuple[Tuple[Participant, Participant], int]]:
        res = []
        for i in range(len(rows)):
            row = rows[i]
            p1 = self.p1[row]
            col = cols[i]
            p2 = self.p2[col]
            res.append(((p1, p2), self.double_arr[row][col]))

        return res
        ...

def divide_up_participants(all_participants : List[Participant]):
    straight_guys = list(filter(lambda participant : participant.sex == "M" and participant.is_straight() and not participant.friend_only, all_participants))
    straight_girls = list(filter(lambda participant : participant.sex == "F" and participant.is_straight() and not participant.friend_only, all_participants))

    lesbians = list(filter(lambda participant : participant.is_lesbian() and not participant.friend_only, all_participants))
    gay_ppl = list(filter(lambda participant : participant.is_gay()and  not participant.friend_only, all_participants))
    friend_only = list(filter(lambda participant : participant.friend_only, all_participants))

    return (straight_guys, straight_girls, lesbians, gay_ppl, friend_only)

def equal_distribution(participants : List[Participant]) -> Tuple[List[Participant], List[Participant]]:
    list1 = []
    list2 = []
    for participant in participants:
        if len(list1) == len(list2):
            list1.append(participant)
        else: 
            list2.append(participant)
    return list1, list2

def matching(participants1 : List[Participant], participants2: List[Participant], 
                 interests : dict[str, dict[Participant, int]], match_type: MatchType, previous_matchings : Set[FrozenSet[ Participant]] = set(), randomize = False, round : int = 0, ) -> List[Matching]:
    # [((Participant(name='buchi', sex='M', wants='F'), Participant(name='Michael Ugochukwu', sex='F', wants='M')), 25)]

    hungarian_graph = {}
    for participant in participants1:
        hungarian_graph[participant] = {}
        for other in participants2:
            hungarian_graph[participant][other] = participant.calculate_match(other=other, interests=interests, randomize=randomize, previous_matchings=previous_matchings)
    inter : Intermediate = Intermediate.from_graph(hungarian_graph)
    rows, cols = linear_sum_assignment(inter.double_arr, maximize=True)
    
    result = inter.parse_matching(rows=rows, cols=cols)
    # result= algorithm.find_matching(hungarian_graph)
    
    matches : List[Matching] = []
    if result == False:
        return []
    for res in result:
        players,  score = res
        p1, p2 = players
        matches.append(Matching(participants=[p1, p2], score=score, round=round, match_type=match_type))
    return matches

def generate_interest_dict( participants : List[ValentineProfile]) -> Tuple[dict[str, dict[Participant, int]], List[Participant]]:
    res = {}
    participants_array = []
    for player in participants:
        participant_structure = Participant.from_valentine_profile(player)
        for interest in player.interests:
            interest_name = interest.name
            slot = res.get(interest_name)
            if slot == None:
                res[interest_name] = {}
                slot = res[interest_name]
            slot[participant_structure] = interest.score
        participants_array.append(participant_structure)
    return res, participants_array

def generate_interest_dict_v2(participants : List[ValentineProfile])->Tuple[dict[str, dict[Participant, int]], List[Participant]]:
    res = {}
    participants_array = []

    for interest in interest_list:
        res[interest] = {}
    for player in participants:
        participant_structure = Participant.from_valentine_profile(player)
        for interest in player.interests:
            interest_name = interest.name
            if interest_name in res:
                res[interest_name][participant_structure] = interest.score

        participants_array.append(participant_structure)
    return res, participants_array
    ...
async def create_profile(input : ValentineProfileInput, info : Info) -> ValentineProfileOut:
    user= info.context.user
    clerk_id = user.clerk_id
    data =await ValentineProfile.find_one(ValentineProfile.clerk_id == clerk_id)
    input_data = input.to_pydantic(clerk_id=clerk_id)
    
    if (data):
        data.name = input_data.name
        data.email = input_data.email
        data.interests = input_data.interests
        data.wants = input_data.wants
        data.gender = input_data.gender
        data.discord = input_data.discord
        data.instagram = input_data.instagram
        data.wants = input_data.wants
        await data.save()
        return ValentineProfileOut.from_pydantic(data)
    else:    
        await input_data.save()
        return ValentineProfileOut.from_pydantic(input_data)

async def db_profile(info : Info) -> ValentineProfile:
    user= info.context.user
    clerk_id = user.clerk_id
    data =await ValentineProfile.find_one(ValentineProfile.clerk_id == clerk_id)
    return data

async def my_profile(info : Info) ->ValentineProfileOut:
    data = await db_profile(info)
    if not data:
        raise Exception("You don't exist")
    return ValentineProfileOut.from_pydantic(data)

def extract_unmatched(all : List[Participant], all_matches : List[Matching]) -> Set[Participant]:
    matches = set()
    unmatched = set()
    for match in all_matches:
        for participant in match.participants:
            matches.add(participant)
    for participant in all:
        if participant not in matches:
            unmatched.add(participant)
    return unmatched

def can_match(p1 : List[Participant], p2 : List[Participant]):
    return len(p1) > 0 and len(p2) > 0

def extract_matching_to_set(matching : List[Matching]) ->Set[FrozenSet[Participant]]:
    result = set()
    for match in matching:
        p1, p2 = match.participants
        result.add(frozenset([p1, p2]))
    return result
    pass

def extract_round(matching : List[Matching]) -> int:
    maximum = 0
    for match in matching:
        maximum = max(maximum, match.round)
    return maximum + 1

async def publish_matching(matches : List[Matching], players : List[ValentineProfile]) :
    store  : dict[str, ValentineProfile] = {}
    for player in players:
        store[player.id] = player
    arr = []
    for match in matches:
        profs : List[ValentineProfile] = []
        for participant in match.participants:
            profs.append(store[participant.id])
        match.valentine_participants = profs
        arr.append(match.save())
        
    await asyncio.gather(*arr)
    print("Match saved")

async def create_matchings():
    print("Matching begin")
    previous_matches = await Matching.find(fetch_links=True, sort=-Matching.round).to_list()
    round=  extract_round(matching=previous_matches)
    match_set= extract_matching_to_set(previous_matches)
    players = await ValentineProfile.find_all().to_list()
    interests_dict, participant_players = generate_interest_dict_v2(players)
    print("Interests generated")
    straight_guys, straight_girls, lesbians, gay_guys, friend_only = divide_up_participants(participant_players)
    straight_matching : List[Matching] = []
    
    if can_match(straight_girls, straight_girls):
        straight_matching = matching(straight_guys, straight_girls, interests_dict, round=round, match_type="Romantic Matching", previous_matchings=match_set)
        print("performed straight matching")
    lesbian_matching : List[Matching] = []
    les1, les2 = equal_distribution(lesbians)
    if can_match(les1, les2 ):
        lesbian_matching = matching(les1, les2, interests_dict,round=round, match_type="Romantic Matching", previous_matchings=match_set)
        print("performed lesbian matching")
    gay_matching  : List[Matching] = []
    gay1, gay2 = equal_distribution(gay_guys)
    if can_match(gay1, gay2):
        gay_matching = matching(gay1, gay2, interests_dict, round=round, match_type="Romantic Matching", previous_matchings=match_set)
        print("performed gay matching")
    all_matches = []
    all_matches.extend(straight_matching)
    all_matches.extend(lesbian_matching)
    all_matches.extend(gay_matching)
    

    unmatched = extract_unmatched(participant_players, all_matches)
    unmatched1, unmatched2 = equal_distribution(list(unmatched))

    bro_matching : List[Matching] = []
    if can_match(unmatched1, unmatched2):
        print("Bro match beginning")
        bro_matching = matching(unmatched1, unmatched2, interests_dict, round=round, match_type="Bro Matching", previous_matchings=match_set)
        print("performed bro matching")

    all_matches.extend(bro_matching)
    await publish_matching(all_matches, players)
    return True
    
async def my_matchings(info : Info)-> List[MatchOut]:
    me = await db_profile(info)
    matches = await Matching.find(In(Matching.valentine_participants.id, [me.id]) ,fetch_links=True).to_list()
    matches.sort(reverse=True, key=lambda x : x.round)
    res = [MatchOut.from_pydantic(match) for match in matches]
    return res
