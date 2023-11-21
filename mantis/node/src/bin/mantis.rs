use bip32::secp256k1::elliptic_curve::rand_core::block;
use cosmos_sdk_proto::{
    cosmos::{auth::v1beta1::BaseAccount, base::v1beta1::Coin},
    cosmwasm::{self, wasm::v1::QuerySmartContractStateRequest},
};
use cosmos_sdk_proto::{traits::Message, Any};
use cosmrs::{
    tendermint::{chain, block::Height},
    tx::{Msg, SignDoc},
};

use cosmrs::{
    cosmwasm::MsgExecuteContract,
    rpc::Client,
    tx::{self, Fee, SignerInfo},
    AccountId,
};
use cw_mantis_order::{OrderItem, OrderSubMsg};
use mantis_node::{
    mantis::{
        args::*,
        cosmos::{client::*, *},
    },
    prelude::*,
};

#[tokio::main]
async fn main() {
    let args = MantisArgs::parsed();
    println!("args: {:?}", args);
    let wasm_read_client = create_wasm_query_client(&args.rpc_centauri).await;

    let signer = mantis_node::mantis::cosmos::signer::from_mnemonic(
        args.wallet.as_str(),
        "m/44'/118'/0'/0/0",
    )
    .expect("mnemonic");

    loop {
        let rpc_client: cosmrs::rpc::HttpClient =
            cosmrs::rpc::HttpClient::new(args.rpc_centauri.as_ref()).unwrap();
        let status = rpc_client.status().await.expect("status").sync_info;
        println!("status: {:?}", status);

        
        let (block, account) = get_latest_block_and_account(&args.rpc_centauri, 
            signer
            .public_key()
            .account_id("centauri")
            .expect("key")
            .to_string(),
        ).await;

        let mut cosmos_query_client = create_cosmos_query_client(&args.rpc_centauri).await;
        print!("client 1");
        let mut write_client = create_wasm_write_client(&args.rpc_centauri).await;
        print!("client 2");

        println!("acc: {:?}", account);
        if let Some(assets) = args.simulate.clone() {
            simulate_order(
                &mut write_client,
                &mut cosmos_query_client,
                args.order_contract.clone(),
                assets,
                &signer,
                account,
                block,
                &args.rpc_centauri,
            )
            .await;
        };
    }
}

/// `assets` - is comma separate list. each entry is amount u64 glued with alphanumeric denomination
/// that is splitted into array of CosmWasm coins.
/// one coin is chosen as given,
/// from remaining 2 other coins one is chosen as wanted
/// amount of count is randomized around values
///
/// `write_client`
/// `order_contract` - orders are formed for give and want, and send to orders contract.
/// timeout is also randomized starting from 10 to 100 blocks
///
/// Also calls `timeout` so old orders are cleaned.
async fn simulate_order(
    write_client: &mut CosmWasmWriteClient,
    cosmos_query_client: &mut CosmosQueryClient,
    order_contract: String,
    asset: String,
    signing_key: &cosmrs::crypto::secp256k1::SigningKey,
    account: BaseAccount,
    block: cosmrs::tendermint::block::Height,
    rpc: &str,
) {
    let coins: Vec<_> = asset
        .split(',')
        .map(|x| cosmwasm_std::Coin::from_str(x).expect("coin"))
        .collect();

    let coins = if std::time::Instant::now().elapsed().as_millis() % 2 == 0 {
        (coins[0].clone(), coins[1].clone())
    } else {
        (coins[1].clone(), coins[0].clone())
    };
    if std::time::Instant::now().elapsed().as_millis() % 1000 == 0 {
        let auth_info = SignerInfo::single_direct(Some(signing_key.public_key()), account.sequence)
            .auth_info(Fee {
                amount: vec![cosmrs::Coin {
                    amount: 10,
                    denom: cosmrs::Denom::from_str("ppica").expect("denom"),
                }],
                gas_limit: 1_000_000,
                payer: None,
                granter: None,
            });

        let msg = cw_mantis_order::ExecMsg::Order {
            msg: OrderSubMsg {
                wants: cosmwasm_std::Coin {
                    amount: coins.0.amount,
                    denom: coins.0.denom.clone(),
                },
                transfer: None,
                timeout: block.value() + 100,
                min_fill: None,
            },
        };
        println!("msg: {:?}", msg);
        let msg = MsgExecuteContract {
            sender: signing_key
                .public_key()
                .account_id("centauri")
                .expect("account"),
            contract: AccountId::from_str(&order_contract).expect("contract"),
            msg: serde_json_wasm::to_vec(&msg).expect("json"),
            funds: vec![cosmrs::Coin {
                amount: coins.1.amount.into(),
                denom: cosmrs::Denom::from_str(&coins.1.denom).expect("denom"),
            }],
        };

        tx_broadcast_single_signed_msg(msg.to_any().expect("proto"), block, auth_info, account, rpc, signing_key).await;

        // here parse contract result for its response
    }
}


/// gets orders, groups by pairs
/// solves them using algorithm
/// if any volume solved, posts solution
///
/// gets data from chain pools/fees on osmosis and neutron
/// gets CVM routing data
/// uses cfmm algorithm
async fn solve(
    read: &mut CosmWasmReadClient,
    _write: CosmWasmWriteClient,
    order_contract: String,
    _cvm_contract: String,
) {
    let query = cw_mantis_order::QueryMsg::GetAllOrders {};
    let orders_request = QuerySmartContractStateRequest {
        address: order_contract.clone(),
        query_data: serde_json_wasm::to_vec(&query).expect("json"),
    };
    let orders = read
        .smart_contract_state(orders_request)
        .await
        .expect("orders obtained")
        .into_inner()
        .data;
    let orders: Vec<OrderItem> = serde_json_wasm::from_slice(&orders).expect("orders");

    let orders = orders.into_iter().group_by(|x| {
        if x.given.denom < x.msg.wants.denom {
            (x.given.denom.clone(), x.msg.wants.denom.clone())
        } else {
            (x.msg.wants.denom.clone(), x.given.denom.clone())
        }
    });
    for (pair, orders) in orders.into_iter() {
        // solve here !
        // post solution
        // just print them for now
        println!("pair {pair:?} orders: {:?}", orders.collect::<Vec<_>>());
    }
}
