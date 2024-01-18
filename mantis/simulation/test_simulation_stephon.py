import itertools
import numpy as np
import cvxpy as cp
import pandas as pd
import random
from enum import Enum
from typing import TypeVar, List, Dict, Tuple, Optional, Union, Callable
from pydantic import BaseModel
from tqdm import tqdm
from datetime import datetime

TAssetId = TypeVar("TAssetId")
TNetworkId = TypeVar("TNetworkId")

class CFMM: 
    
    def __init__(self, tokens: Tuple[str, str], reserves: np.array, tx_cost: float, pool_fee: float, is_transfer: bool) -> None:
        self.tokens = tokens 
        self.reserves = reserves
        self.pool_fee = pool_fee
        self.is_transfer = is_transfer
        self.tx_cost = tx_cost
        
    def __len__(self) -> int:
        return len(self.tokens)
    
    def __str__(self) -> str: 
        return f'CFMM({self.tokens}, {self.reserves}, {self.tx_cost}, {self.pool_fee}, {self.is_transfer})'
    
    def __repr__(self) -> str:
        return str(self)

def flip(p: float) -> bool:
    return random.random() < p

def populate_chain_dict(chains: dict[TNetworkId, list[TAssetId]], center_node: TNetworkId):
    # Add tokens with denom to Center Node
    # Basic IBC transfer
    for chain, tokens in chains.items():
        if chain != center_node:
            chains[center_node].extend(f"{chain}/{token}" for token in tokens)

    1# Add tokens from Center Node to outers
    
    # Simulate IBC transfer through Composable Cosmos
    for chain, tokens in chains.items():
        if chain != center_node:
            chains[chain].extend(
                f"{center_node}/{token}"
                for token in chains[center_node]
                if f"{chain}/" not in token
            )
            
def get_mapping_matrices(all_tokens, all_cfmms): 
    count_tokens = len(all_tokens)
    
    mapping_matrices = [] 
    for cfmm in all_cfmms: 
        n_i = len(cfmm.tokens)
        A_i = np.zeros((count_tokens, n_i))
        for i, token in enumerate(cfmm.tokens):
            A_i[all_tokens.index(token), i] = 1
        mapping_matrices.append(A_i)
        
    return mapping_matrices

def solve_with_unknown_eta(origin_token: str, number_of_init_tokens: float, obj_token: str, all_tokens, all_cfmms, mapping_matrices):
    """In this function we solve the optimal routing problem with unknown eta values. These values are either 0 or 1. 
    How we approach this is by not including any CFMMs into the constraints that have an 
    eta value equal to 0. This way, we can solve the problem with CVXPY and not run into numerical issues. 

    Args:
        origin_token (str): The origin token
        number_of_init_tokens (float): The number of tokens we start with
        obj_token (str): The token we want to end up with
        all_tokens (_type_): A list of all tokens in the simulation
        all_cfmms (_type_): A list of all possible CFMMs in the simulation
        mapping_matrices (_type_): Matrices which map the tokens in local indices for each CFMM to global indices
    """
    # Build local-global matrices
    count_tokens = len(all_tokens)
    current_assets = np.zeros(count_tokens)  # Initial assets
    current_assets[all_tokens.index(origin_token)] = number_of_init_tokens
    
    # Getting the cfmms which have been chosen and their mapping matrices
    all_reserves = [cfmm.reserves for cfmm in all_cfmms]
    all_fees = [cfmm.pool_fee for cfmm in all_cfmms]
    all_tx_cost = [cfmm.tx_cost for cfmm in all_cfmms]
    
    deltas = [cp.Variable(len(cfmm), nonneg=True) for cfmm in all_cfmms] 
    lambdas = [cp.Variable(len(cfmm), nonneg=True) for cfmm in all_cfmms]
    
    eta = cp.Variable(len(all_cfmms), nonneg=True)
    
    psi = cp.sum([A_i @ (LAMBDA - DELTA) for A_i, DELTA, LAMBDA in zip(mapping_matrices, deltas, lambdas)])
    
    # Creating the objective 
    objective = cp.Maximize(psi[all_tokens.index(obj_token)] - cp.sum([eta[i] * all_tx_cost[i] for i in range(len(all_cfmms))]))
    
    # Creating the reserves
    new_reserves_list = [
        R + gamma * D - L for R, gamma, D, L in zip(all_reserves, all_fees, deltas, lambdas)
    ]
    
    # Creating constraints 
    constraints = [psi + current_assets >= 0]
    
    # Pool constraints
    for i, cfmm in enumerate(all_cfmms):
        new_reserves = new_reserves_list[i]
        old_reserves = all_reserves[i]
        if cfmm.is_transfer:
            constraints.append(cp.sum(new_reserves) >= cp.sum(old_reserves))
            constraints.append(new_reserves >= 0)
        else:
            # In this case we don't have a transfer pool but a regular swap
            constraints.append(cp.geo_mean(new_reserves) >= cp.geo_mean(old_reserves))
            
    for i in range(len(all_cfmms)):
        constraints.append(deltas[i] <= MAX_RESERVE * eta[i])
        
    # Set up and solve the problem
    problem = cp.Problem(objective, constraints)
    
    problem.solve(verbose=True, solver='CLARABEL', qcp=False)
    
    return deltas, lambdas, psi, eta

def solve_with_known_eta(origin_token: str, number_of_init_tokens: float, obj_token: str, all_tokens, all_cfmms, mapping_matrices, eta): 
    """In this function we solve the optimal routing problem with known eta values. These values are either 0 or 1. 
    How we approach this is by not including any CFMMs into the constraints that have an 
    eta value equal to 0. This way, we can solve the problem with CVXPY and not run into numerical issues. 

    Args:
        origin_token (str): The origin token
        number_of_init_tokens (float): The number of tokens we start with
        obj_token (str): The token we want to end up with
        all_tokens (_type_): A list of all tokens in the simulation
        all_cfmms (_type_): A list of all possible CFMMs in the simulation
        mapping_matrices (_type_): Matrices which map the tokens in local indices for each CFMM to global indices
        eta (_type_): An array of boolean values indicating whether a CFMM is used or not
    """
    # Build local-global matrices
    count_tokens = len(all_tokens)
    current_assets = np.zeros(count_tokens)  # Initial assets
    current_assets[all_tokens.index(origin_token)] = number_of_init_tokens
    
    # Getting the cfmms which have been chosen and their mapping matrices
    all_reserves = [cfmm.reserves for cfmm in all_cfmms]
    all_fees = [cfmm.pool_fee for cfmm in all_cfmms]
    all_tx_cost = [cfmm.tx_cost for cfmm in all_cfmms]
    
    cfmm_tx_cost = sum([eta[i] * all_tx_cost[i] for i in range(len(all_cfmms))])
    
    deltas = [cp.Variable(len(cfmm), nonneg=True) for cfmm in all_cfmms] 
    lambdas = [cp.Variable(len(cfmm), nonneg=True) for cfmm in all_cfmms]
    
    psi = cp.sum([A_i @ (LAMBDA - DELTA) for A_i, DELTA, LAMBDA in zip(mapping_matrices, deltas, lambdas)])
    
    # Creating the objective 
    objective = cp.Maximize(psi[all_tokens.index(obj_token)] - cfmm_tx_cost)
    
    # Creating the reserves
    new_reserves_list = [
        R + gamma * D - L for R, gamma, D, L in zip(all_reserves, all_fees, deltas, lambdas)
    ]
    
    # Creating the constraints
    constraints = [psi + current_assets >= 0] 
    
    # Pool constraints 
    for i, cfmm in enumerate(all_cfmms): 
        new_reserves = new_reserves_list[i]
        old_reserves = all_reserves[i]
        if cfmm.is_transfer: 
            constraints.append(cp.sum(new_reserves) >= cp.sum(old_reserves))
            constraints.append(new_reserves >= 0)
            
        else: 
            # In this case we don't have a transfer pool but a regular swap
            constraints.append(cp.geo_mean(new_reserves) >= cp.geo_mean(old_reserves))
            
    # Forcing delta and lambda to be 0 if eta is 0
    for i, cfmm in enumerate(all_cfmms): 
        if eta[i] == 0: 
            constraints.append(deltas[i] == 0)
            constraints.append(lambdas[i] == 0)
        else: 
            constraints.append(deltas[i] <= MAX_RESERVE)
    
    # Set up and solve the problem 
    problem = cp.Problem(objective, constraints)
    
    problem.solve(verbose=True, solver='CLARABEL', qcp=False)
    
    return deltas, lambdas, psi

    
if __name__ == '__main__': 
    CENTER_NODE = "CENTAURI"  # Name of center Node
    ORIGIN_TOKEN = "WETH"
    OBJ_TOKEN = "ATOM"

    MAX_RESERVE = 1e10
    
    CENTER_NODE = "CENTAURI"  # Name of center Node
    ORIGIN_TOKEN = "WETH"
    OBJ_TOKEN = "ATOM"

    chains: dict[str, list[str]] = {
        "ETHEREUM": ["WETH", "USDC", "SHIBA"],
        CENTER_NODE: [],
        "OSMOSIS": ["ATOM","SCRT"],
    }

    populate_chain_dict(chains,CENTER_NODE)
    
    all_tokens = []
    all_cfmm_combinations = []
    all_cfmms = []
    
    for other_chain, other_tokens in chains.items():
        
        all_tokens.extend(other_tokens)
        all_cfmm_combinations.extend(itertools.combinations(other_tokens, 2))
    
    # Creating the CFMMs that are not transfers
    for cfmm in all_cfmm_combinations: 
        random_reserves = np.random.uniform(9500, 10051, 2)
        random_tx_cost = np.random.uniform(0, 20) 
        random_pool_fee = np.random.uniform(0.97, 0.99)
        mm = CFMM(cfmm, random_reserves, random_tx_cost, random_pool_fee, False)
        all_cfmms.append(mm)
    
    # Creating the CFMMs that are transfers
    for token_on_center in chains[CENTER_NODE]:
        for other_chain, other_tokens in chains.items():
            if other_chain != CENTER_NODE:
                for other_token in other_tokens:
                    # Check wether the chain has the token in center, or the other way around
                    # Could cause problems if chainName == tokensName (for example OSMOSIS)
                    if other_token in token_on_center or token_on_center in other_token:
                        random_reserves = np.random.uniform(9500, 10051, 2)
                        random_tx_cost = np.random.uniform(0, 20) 
                        random_pool_fee = np.random.uniform(0.97, 0.99)
                        cfmm = CFMM((token_on_center, other_token), random_reserves, random_tx_cost, random_pool_fee, True)
                        all_cfmms.append(cfmm)
                        
    # Now, we get into the simulation
    # Create a random eta vector
    eta = [flip(0.5) for _ in range(len(all_cfmms))]
    
    # Get the mapping matrices
    mapping_matrices = get_mapping_matrices(all_tokens, all_cfmms)
    
    # Solve the problem
    initial_tokens = 2000
    deltas, lambdas, psi = solve_with_known_eta(ORIGIN_TOKEN, initial_tokens, OBJ_TOKEN, 
                                                all_tokens, all_cfmms, mapping_matrices, 
                                                eta)
    
    print(deltas, lambdas, psi)

