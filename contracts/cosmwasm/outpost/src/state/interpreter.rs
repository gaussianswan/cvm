use cosmwasm_std::{Deps, StdResult};
use cvm_runtime::ExecutorOrigin;
use cw_storage_plus::Item;

use crate::prelude::*;

pub type InterpreterId = cvm_runtime::shared::Displayed<u128>;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "json-schema", // all(feature = "json-schema", not(target_arch = "wasm32")),
    derive(schemars::JsonSchema)
)]
pub(crate) struct Interpreter {
    /// contract address
    pub address: Addr,
    pub interpreter_id: InterpreterId,
}

pub(crate) fn get_by_origin(deps: Deps, origin: ExecutorOrigin) -> StdResult<Interpreter> {
    let id = INTERPRETERS_ORIGIN_TO_ID.load(deps.storage, origin)?;
    INTERPRETERS.load(deps.storage, id)
}

pub(crate) const INTERPRETERS_COUNT: Item<u128> = Item::new("interpreter_count");

pub(crate) const INTERPRETERS_ORIGIN_TO_ID: Map<ExecutorOrigin, u128> =
    Map::new("interpreters_origin_to_id");

pub(crate) const INTERPRETERS: Map<u128, Interpreter> = Map::new("interpreters");
