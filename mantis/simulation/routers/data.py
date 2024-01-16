# for alignment on input and output of algorithm
import pandas as pd
from enum import Enum
from typing import TypeVar
from pydantic import BaseModel
from strictly_typed_pandas import DataSet

TAssetId = TypeVar("TAssetId")
TNetworkId = TypeVar("TNetworkId")

class AssetTransfers(BaseModel):
    # positive whole numbers, set key
    in_asset_id: str
    out_asset_id: str 
    
    # this is positive whole number too
    # if it is hard if None, please fail if it is None - for now will be always some
    usd_fee_transfer: int | None
    
    # do not care
    metadata: str | None 
    def __init__(self, in_asset_id, out_asset_id, usd_fee_transfer, metadata):
        self.in_asset_id = in_asset_id
        self.out_asset_id = out_asset_id
        self.metadata = metadata
        self.usd_fee_transfer = usd_fee_transfer
    
# pool are bidirectional, so so in can be out and other way
class AssetPairsXyk(BaseModel):
    # set key
    pool_id: int
    in_asset_id: int
    out_asset_id: int
    
    fee_of_in_per_million: int
    fee_of_out_per_million: int 
    weight_of_a: int 
    weight_of_b: int 
    # if it is hard if None, please fail if it is None - for now will be always some
    pool_value_in_usd: int | None
    
    # total amounts in reserves R
    in_token_amount: int
    out_token_amount: int
    
    metadata: str | None
    
# this is what user asks for
class Input(BaseModel):
    # natural set key is ordered pair (in_token_id, out_token_id)
    in_token_id: int
    out_token_id: int
    # tendered amount DELTA
    in_amount: int
    # expected received amount LAMBDA
    out_amount: int
    # if max is True, user wants to spent all in to get at least out
    # if max is False, user wants to get exact out, but spent as small as possible in
    # please fail if bool is False for now
    max: bool

    
class SingleInputAssetCvmRoute(BaseModel):
    pass    


# transfer assets
class Spawn(BaseModel):
    # amount to take with transfer
    # None means all (DELTA)
    in_asset_amount: int | None
    out_asset_id: int
    next: SingleInputAssetCvmRoute

class Exchange(BaseModel):
    # none means all (DELTA)
    in_asset_amount: int | None
    pool_id : int
    next: SingleInputAssetCvmRoute

# always starts with Input amount and asset
class SingleInputAssetCvmRoute(BaseModel):
    next: list[Exchange | Spawn]

SingleInputAssetCvmRoute.update_forward_refs()

class SolutionType(Enum):
    # really to find any solution
    FAILED = 0
    # all assets will be solved with limit
    FULL = 1
    # will be solved under desired limit
    UNDER_LIMIT = 2
    # will solve within limits, bat only part of assets
    PARTIAL = 3
    

class Output(BaseModel):
    # str describing failure to find any solution
    route: SingleInputAssetCvmRoute | str
    solution_type: SolutionType     

T = TypeVar("T")

class PydanticDataSet(BaseModel, DataSet[T]):
    pass

# global labelling of assets and exchanges
class AllData(BaseModel):
    # DataSet inherits from DataFrame
    # If key is in first set, it cannot be in second set, and other way around
    asset_transfers : PydanticDataSet[AssetTransfers]
    asset_pairs_xyk : PydanticDataSet[AssetPairsXyk]
    # if None, than solution must not contain any joins after forks
    # so A was split into B and C, and then B and C were moved to be D
    # D must "summed" from 2 amounts must be 2 separate routes branches
    fork_joins : str | None
    

# helpers to setup tests data

def test_all_data() -> AllData:
    asset_transfers =  PydanticDataSet[AssetTransfers](pd.read_csv("asset_transfers.csv"))
    assets_pairs_xyk=  PydanticDataSet[AssetPairsXyk](pd.read_csv("assets_pairs_xyk.csv"))
    return AllData(assets_pairs_xyk, asset_transfers)


def new_input(in_token_id, out_token_id, in_amount, out_amount):
    return Input(in_token_id = in_token_id, out_token_id = out_token_id, in_amount = in_amount, out_amount = out_amount, max = True)     

def new_pair(pool_id, in_asset_id, out_asset_id, fee_of_in_per_million, fee_of_out_per_million, weight_of_a, weight_of_b, pool_value_in_usd, in_token_amount, out_token_amount, metadata):
    return AssetPairsXyk(pool_id = pool_id, in_asset_id = in_asset_id, out_asset_id = out_asset_id, fee_of_in_per_million = fee_of_in_per_million, fee_of_out_per_million = fee_of_out_per_million, weight_of_a = weight_of_a, weight_of_b = weight_of_b, pool_value_in_usd = pool_value_in_usd, in_token_amount = in_token_amount, out_token_amount = out_token_amount, metadata = metadata)
    
