# generated by datamodel-codegen:
#   filename:  execute.json

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Extra, Field, conint


class Addr(BaseModel):
    __root__: str = Field(
        ...,
        description="A human readable address.\n\nIn Cosmos, this is typically bech32 encoded. But for multi-chain smart contracts no assumptions should be made other than being UTF-8 encoded and of reasonable length.\n\nThis type represents a validated address. It can be created in the following ways 1. Use `Addr::unchecked(input)` 2. Use `let checked: Addr = deps.api.addr_validate(input)?` 3. Use `let checked: Addr = deps.api.addr_humanize(canonical_addr)?` 4. Deserialize from JSON. This must only be done from JSON that was validated before such as a contract's state. `Addr` must not be used in messages sent by the user because this would result in unvalidated instances.\n\nThis type is immutable. If you really need to mutate it (Really? Are you sure?), create a mutable copy using `let mut mutable = Addr::to_string()` and operate on that `String` instance.",
    )


class Adr08IbcCallbacks(BaseModel):
    """
    if chain has IBC SDK callbacks enabled
    """


class AssetId(BaseModel):
    __root__: str = Field(
        ...,
        description='Newtype for XCVM assets ID. Must be unique for each asset and must never change. This ID is an opaque, arbitrary type from the XCVM protocol and no assumption must be made on how it is computed.',
    )


class Native(BaseModel):
    denom: str


class AssetReference1(BaseModel):
    """
    Definition of an asset native to some chain to operate on. For example for Cosmos CW and EVM chains both CW20 and ERC20 can be actual. So if asset is local or only remote to some chain depends on context of network or connection. this design leads to some dummy matches, but in general unifies code (so that if one have to solve other chain route it can)
    """

    class Config:
        extra = Extra.forbid

    native: Native


class Cw20(BaseModel):
    contract: Addr


class AssetReference2(BaseModel):
    """
    Definition of an asset native to some chain to operate on. For example for Cosmos CW and EVM chains both CW20 and ERC20 can be actual. So if asset is local or only remote to some chain depends on context of network or connection. this design leads to some dummy matches, but in general unifies code (so that if one have to solve other chain route it can)
    """

    class Config:
        extra = Extra.forbid

    cw20: Cw20


class AssetReference(BaseModel):
    __root__: Union[AssetReference1, AssetReference2] = Field(
        ...,
        description='Definition of an asset native to some chain to operate on. For example for Cosmos CW and EVM chains both CW20 and ERC20 can be actual. So if asset is local or only remote to some chain depends on context of network or connection. this design leads to some dummy matches, but in general unifies code (so that if one have to solve other chain route it can)',
    )


class BindingValue2(BaseModel):
    """
    Asset's address
    """

    class Config:
        extra = Extra.forbid

    asset: AssetId


class Local(BaseModel):
    user: Addr


class CallOrigin2(BaseModel):
    """
    The Origin that executed the XCVM operation. Origin was verified to satisfy security semantics for execution.
    """

    class Config:
        extra = Extra.forbid

    Local: Local


class ChannelId(BaseModel):
    __root__: str


class ForceRemoveAsset(BaseModel):
    asset_id: AssetId


class ConfigSubMsg6(BaseModel):
    """
    Message sent by an admin to remove an asset from registry.
    """

    class Config:
        extra = Extra.forbid

    force_remove_asset: ForceRemoveAsset


class ForceInstantiate(BaseModel):
    salt: Optional[str] = None
    user_origin: Addr


class ConfigSubMsg8(BaseModel):
    """
    instantiates default interpreter on behalf of user `salt` - human string, converted to hex or base64 depending on implementation
    """

    class Config:
        extra = Extra.forbid

    force_instantiate: ForceInstantiate


class ConnectionId(BaseModel):
    __root__: str


class DestinationForXcAddr1(Enum):
    tip = 'tip'


class DisplayedForUint128(BaseModel):
    __root__: str = Field(
        ...,
        description='A wrapper around a type which is serde-serialised as a string.\n\nFor serde-serialisation to be implemented for the type `T` must implement `Display` and `FromStr` traits.\n\n```rust use cvm::shared::Displayed;\n\n#[derive(serde::Serialize, serde::Deserialize)] struct Foo { value: Displayed<u64> }\n\nlet encoded = serde_json_wasm::to_string(&Foo { value: Displayed(42) }).unwrap(); assert_eq!(r#"{"value":"42"}"#, encoded);\n\nlet decoded = serde_json_wasm::from_str::<Foo>(r#"{"value":"42"}"#).unwrap(); assert_eq!(Displayed(42), decoded.value); ```',
    )


class DisplayedForUint64(BaseModel):
    __root__: str = Field(
        ...,
        description='A wrapper around a type which is serde-serialised as a string.\n\nFor serde-serialisation to be implemented for the type `T` must implement `Display` and `FromStr` traits.\n\n```rust use cvm::shared::Displayed;\n\n#[derive(serde::Serialize, serde::Deserialize)] struct Foo { value: Displayed<u64> }\n\nlet encoded = serde_json_wasm::to_string(&Foo { value: Displayed(42) }).unwrap(); assert_eq!(r#"{"value":"42"}"#, encoded);\n\nlet decoded = serde_json_wasm::from_str::<Foo>(r#"{"value":"42"}"#).unwrap(); assert_eq!(Displayed(42), decoded.value); ```',
    )


class OsmosisPoolManagerModuleV1Beta1(BaseModel):
    pool_id: conint(ge=0)
    token_a: str
    token_b: str


class ExchangeType1(BaseModel):
    class Config:
        extra = Extra.forbid

    osmosis_pool_manager_module_v1_beta1: OsmosisPoolManagerModuleV1Beta1


class AstroportRouterContract(BaseModel):
    address: Addr
    token_a: str
    token_b: str


class ExchangeType2(BaseModel):
    class Config:
        extra = Extra.forbid

    astroport_router_contract: AstroportRouterContract


class ExchangeType(BaseModel):
    __root__: Union[ExchangeType1, ExchangeType2]


class FundsForDisplayedForUint128(BaseModel):
    """
    a set of assets with non zero balances
    """

    __root__: List[List[Union[AssetId, DisplayedForUint128]]] = Field(
        ..., description='a set of assets with non zero balances'
    )


class IbcEndpoint(BaseModel):
    channel_id: str
    port_id: str


class IbcIcs20Sender(Enum):
    CosmosStargateIbcApplicationsTransferV1MsgTransfer = (
        'CosmosStargateIbcApplicationsTransferV1MsgTransfer'
    )
    CosmWasmStd1_3 = 'CosmWasmStd1_3'


class IcsPair(BaseModel):
    """
    we need both, so we can unwrap
    """

    sink: ChannelId
    source: ChannelId


class Binding(BaseModel):
    __root__: conint(ge=0)


class NetworkId(BaseModel):
    __root__: conint(ge=0) = Field(
        ...,
        description='Newtype for XCVM networks ID. Must be unique for each network and must never change. This ID is an opaque, arbitrary type from the XCVM protocol and no assumption must be made on how it is computed.',
    )


class OsmosisIbcHooks(BaseModel):
    callback: bool


class CosmWasm(BaseModel):
    admin: Addr = Field(..., description='admin of everything')
    contract: Addr
    interpreter_code_id: conint(ge=0) = Field(
        ..., description='CVM interpreter contract code'
    )


class OutpostId1(BaseModel):
    """
    when message is sent to other side, we should identify receiver of some kind
    """

    class Config:
        extra = Extra.forbid

    cosm_wasm: CosmWasm


class OutpostId(BaseModel):
    __root__: OutpostId1 = Field(
        ...,
        description='when message is sent to other side, we should identify receiver of some kind',
    )


class PFM(BaseModel):
    pass


class Prefix1(BaseModel):
    """
    given prefix you may form accounts from 32 bit addresses or partially identify chains
    """

    class Config:
        extra = Extra.forbid

    s_s58: conint(ge=0)


class Prefix2(BaseModel):
    """
    given prefix you may form accounts from 32 bit addresses or partially identify chains
    """

    class Config:
        extra = Extra.forbid

    bech: str


class Prefix(BaseModel):
    __root__: Union[Prefix1, Prefix2] = Field(
        ...,
        description='given prefix you may form accounts from 32 bit addresses or partially identify chains',
    )


class PrefixedDenom(BaseModel):
    """
    A type that contains the base denomination for ICS20 and the source tracing information path.
    """

    base_denom: str = Field(
        ..., description='Base denomination of the relayed fungible token.'
    )
    trace_path: str = Field(
        ...,
        description='A series of `{port-id}/{channel-id}`s for tracing the source of the token.',
    )


class Register1(Enum):
    """
    Instruction pointer
    """

    ip = 'ip'


class Register2(Enum):
    """
    Tip's address
    """

    tip = 'tip'


class Register3(Enum):
    """
    Interpreter's address
    """

    this = 'this'


class Register4(Enum):
    """
    Result of the last executed instruction. If not empty, program did not executed to the end.
    """

    result = 'result'


class Register5(BaseModel):
    """
    Refers to amount transferred via Spawn or originating call
    """

    class Config:
        extra = Extra.forbid

    carry: AssetId


class Register(BaseModel):
    __root__: Union[Register1, Register2, Register3, Register4, Register5]


class RelativeTimeout1(BaseModel):
    """
    Timeout is relative to the current block timestamp of counter party
    """

    class Config:
        extra = Extra.forbid

    seconds: conint(ge=0)


class RelativeTimeout(BaseModel):
    __root__: RelativeTimeout1 = Field(
        ...,
        description='relative timeout to CW/IBC-rs time. very small, assumed messages are arriving fast enough, like less than hours',
    )


class Uint128(BaseModel):
    __root__: str = Field(
        ...,
        description='A thin wrapper around u128 that is using strings for JSON encoding/decoding, such that the full u128 range can be used for clients that convert JSON numbers to floats, like JavaScript and jq.\n\n# Examples\n\nUse `from` to create instances of this and `u128` to get the value out:\n\n``` # use cosmwasm_std::Uint128; let a = Uint128::from(123u128); assert_eq!(a.u128(), 123);\n\nlet b = Uint128::from(42u64); assert_eq!(b.u128(), 42);\n\nlet c = Uint128::from(70u32); assert_eq!(c.u128(), 70); ```',
    )


class UserId(BaseModel):
    __root__: str = Field(
        ...,
        description='Arbitrary `User` type that represent the identity of a user on a given network, usually a public key.',
    )


class UserOrigin(BaseModel):
    """
    The origin of a user, which consist of the composite, origin network and origin network user id.
    """

    network_id: NetworkId
    user_id: UserId


class XcAddr(BaseModel):
    __root__: str = Field(
        ...,
        description='A wrapper around any address on any chain. Similar to `ibc_rs::Signer`(multi encoding), but not depend on ibc code bloat. Unlike parity MultiLocation/Account32/Account20 which hard codes enum into code. Better send canonical address to each chain for performance, But it will also decode/reencode best effort. Inner must be either base64 or hex encoded or contain only characters from these. Added with helper per chain to get final address to use.',
    )


class Amount(BaseModel):
    """
    See https://en.wikipedia.org/wiki/Linear_equation#Slope%E2%80%93intercept_form_or_Gradient-intercept_form
    """

    intercept: Optional[DisplayedForUint128] = Field(
        None, description='absolute amount, optional, default is 0'
    )
    slope: Optional[DisplayedForUint64] = Field(
        None,
        description='part of `MAX_PARTS` from remaining after intercept subtraction, optional, default is 0',
    )


class AssetToNetwork(BaseModel):
    other_asset: AssetId
    other_network: NetworkId
    this_asset: AssetId


class BindingValue1(BaseModel):
    class Config:
        extra = Extra.forbid

    register_: Register = Field(..., alias='register')


class BindingValue3(BaseModel):
    class Config:
        extra = Extra.forbid

    asset_amount: List[Union[AssetId, Amount]] = Field(..., max_items=2, min_items=2)


class BindingValue(BaseModel):
    __root__: Union[BindingValue1, BindingValue2, BindingValue3]


class Remote(BaseModel):
    user_origin: UserOrigin


class CallOrigin1(BaseModel):
    """
    The Origin that executed the XCVM operation. Origin was verified to satisfy security semantics for execution.
    """

    class Config:
        extra = Extra.forbid

    Remote: Remote


class CallOrigin(BaseModel):
    __root__: Union[CallOrigin1, CallOrigin2] = Field(
        ...,
        description='The Origin that executed the XCVM operation. Origin was verified to satisfy security semantics for execution.',
    )


class ChannelInfo(BaseModel):
    """
    Information associated with an IBC channel.
    """

    connection_id: ConnectionId = Field(
        ...,
        description='the connection this exists on (you can use to query client/consensus info)',
    )
    counterparty_endpoint: IbcEndpoint = Field(
        ..., description='the remote channel/port we connect to'
    )
    id: ChannelId = Field(..., description='id of this channel')


class ConfigSubMsg4(BaseModel):
    """
    cross cross chain routing requires a lot of configuration, about chain executing this contract, about connectivity to and of other chains (even if not connected directly) and about assets and services on these chains (in future block hooks and some set of host extensions/precompiles would help to get some info automatically) `Force` message sets the data unconditionally.
    """

    class Config:
        extra = Extra.forbid

    force_asset_to_network_map: AssetToNetwork


class DestinationForXcAddr2(BaseModel):
    class Config:
        extra = Extra.forbid

    account: XcAddr


class DestinationForXcAddr(BaseModel):
    __root__: Union[DestinationForXcAddr1, DestinationForXcAddr2]


class ExchangeItem(BaseModel):
    """
    allows to execute Exchange instruction
    """

    exchange: ExchangeType
    exchange_id: DisplayedForUint128
    network_id: NetworkId


class ExecutorOrigin(BaseModel):
    """
    The interpreter origin, composite of a user origin and a salt.
    """

    salt: str
    user_origin: UserOrigin


class ForeignAssetId1(BaseModel):
    class Config:
        extra = Extra.forbid

    ibc_ics20: PrefixedDenom


class ForeignAssetId(BaseModel):
    __root__: ForeignAssetId1


class FundsForAmount(BaseModel):
    """
    a set of assets with non zero balances
    """

    __root__: List[List[Union[AssetId, Amount]]] = Field(
        ..., description='a set of assets with non zero balances'
    )


class Ics20Features(BaseModel):
    """
    what features/modules/version enabled/installed/configured
    """

    ibc_callbacks: Optional[Adr08IbcCallbacks] = None
    pfm: Optional[PFM] = None
    wasm_hooks: Optional[OsmosisIbcHooks] = Field(
        None, description='if it is exists, chain has that enabled'
    )


class Transfer(BaseModel):
    assets: FundsForAmount
    to: DestinationForXcAddr


class InstructionForArrayOfUint8AndXcAddrAndFundsForAmount1(BaseModel):
    """
    Transfer some [`Assets`] from the current program to the [`to`] account.
    """

    class Config:
        extra = Extra.forbid

    transfer: Transfer


class Call(BaseModel):
    bindings: List[List[Union[Binding, BindingValue]]]
    encoded: List[conint(ge=0)]


class InstructionForArrayOfUint8AndXcAddrAndFundsForAmount2(BaseModel):
    """
    Arbitrary payload representing a raw call inside the current network.

    On picasso, this will be a SCALE encoded dispatch call. On ethereum, an ethereum ABI encoded call. On cosmos, a raw json WasmMsg call.

    Depending on the network, the payload might be more structured than the base call. For most of the network in fact, we need to provide the target address along the payload, which can be encoded inside this single payload.
    """

    class Config:
        extra = Extra.forbid

    call: Call


class Exchange(BaseModel):
    exchange_id: DisplayedForUint128
    give: FundsForAmount
    want: FundsForAmount


class InstructionForArrayOfUint8AndXcAddrAndFundsForAmount4(BaseModel):
    """
    Base XCVM instructions. This set will remain as small as possible, expressiveness must come on `top` of the base instructions.
    """

    class Config:
        extra = Extra.forbid

    exchange: Exchange


class OtherNetworkItem(BaseModel):
    counterparty_timeout: RelativeTimeout = Field(
        ..., description='default timeout to use for direct send'
    )
    ics27_channel: Optional[ChannelInfo] = Field(
        None, description='if there is ICS27 IBC channel opened'
    )
    ics_20: Optional[IcsPair] = None
    use_shortcut: Optional[bool] = Field(
        None,
        description='if true, than will use shortcuts for example, if program transfer only program will just use native transfer or if connection supports exchange, it will use exchange default is false if target chain has CVM gateway',
    )


class Transfer1(BaseModel):
    amount: Uint128
    asset_id: AssetId = Field(..., description='assets from there')
    network: NetworkId = Field(
        ...,
        description='target network, can hope over several networks if route is stored in state',
    )
    receiver: Optional[str] = Field(None, description='by default receiver is this')


class ShortcutSubMsg1(BaseModel):
    class Config:
        extra = Extra.forbid

    transfer: Transfer1


class ShortcutSubMsg(BaseModel):
    __root__: ShortcutSubMsg1


class ExecuteMsg5(BaseModel):
    """
    simple permissionless message which produce xcvm program to test routes
    """

    class Config:
        extra = Extra.forbid

    shortcut: ShortcutSubMsg


class BridgeAsset(BaseModel):
    location_on_network: ForeignAssetId


class ConfigSubMsg5(BaseModel):
    """
    cross cross chain routing requires a lot of configuration, about chain executing this contract, about connectivity to and of other chains (even if not connected directly) and about assets and services on these chains (in future block hooks and some set of host extensions/precompiles would help to get some info automatically) `Force` message sets the data unconditionally.
    """

    class Config:
        extra = Extra.forbid

    force_exchange: ExchangeItem


class Ics20Channel(BaseModel):
    features: Optional[Ics20Features] = None
    sender: IbcIcs20Sender = Field(
        ..., description='specific per chain way to send IBC ICS 20 assets'
    )


class NetworkToNetworkItem(BaseModel):
    from_network_id: NetworkId
    to_network: OtherNetworkItem = Field(
        ..., description='how to send `to_network_id` chain'
    )
    to_network_id: NetworkId


class AssetItem(BaseModel):
    asset_id: AssetId
    bridged: Optional[BridgeAsset] = Field(
        None,
        description='if asset was bridged, it would have way to identify bridge/source/channel',
    )
    local: AssetReference
    network_id: NetworkId = Field(
        ..., description='network id on which this asset id can be used locally'
    )


class ConfigSubMsg2(BaseModel):
    """
    Sets network to network connectivity/routing information
    """

    class Config:
        extra = Extra.forbid

    force_network_to_network: NetworkToNetworkItem


class ConfigSubMsg3(BaseModel):
    """
    Permissioned message (gov or admin) to force set asset information.
    """

    class Config:
        extra = Extra.forbid

    force_asset: AssetItem


class IbcChannels(BaseModel):
    ics20: Optional[Ics20Channel] = None


class IbcEnabled(BaseModel):
    channels: Optional[IbcChannels] = None


class NetworkItem(BaseModel):
    accounts: Optional[Prefix] = Field(None, description='Account encoding type')
    ibc: Optional[IbcEnabled] = None
    network_id: NetworkId
    outpost: Optional[OutpostId] = Field(
        None,
        description='something which will be receiver on other side case of network has XCVM deployed as contract, account address is stored here',
    )


class ConfigSubMsg1(BaseModel):
    """
    Permissioned message (gov or admin) to force set information about network contract is executed. Network can be any network or this network (so it overrides some this network parameters too)
    """

    class Config:
        extra = Extra.forbid

    force_network: NetworkItem


class ExecuteMsg1(BaseModel):
    class Config:
        extra = Extra.forbid

    config: ConfigSubMsg


class ExecuteMsg2(BaseModel):
    """
    Sent by the user to execute a program on their behalf.
    """

    class Config:
        extra = Extra.forbid

    execute_program: ExecuteProgramMsgForNullableFundsForDisplayedForUint128


class ExecuteProgramPrivileged(BaseModel):
    call_origin: CallOrigin = Field(..., description='The origin of the call.')
    execute_program: ExecuteProgramMsgForFundsForDisplayedForUint128 = Field(
        ..., description='Program to execute.'
    )


class ExecuteMsg3(BaseModel):
    """
    Request to execute a program on behalf of given user.

    This can only be sent by trusted contract.  The message is
    """

    class Config:
        extra = Extra.forbid

    execute_program_privileged: ExecuteProgramPrivileged


class ExecuteMsg4(BaseModel):
    """
    Message sent from interpreter trying to spawn program on another network.
    """

    class Config:
        extra = Extra.forbid

    bridge_forward: BridgeForwardMsg


class ExecuteMsg6(BaseModel):
    """
    executed by host as part of memo handling
    """

    class Config:
        extra = Extra.forbid

    message_hook: XcMessageData


class ExecuteMsg(BaseModel):
    __root__: Union[
        ExecuteMsg1, ExecuteMsg2, ExecuteMsg3, ExecuteMsg4, ExecuteMsg5, ExecuteMsg6
    ] = Field(..., title='ExecuteMsg')


class BridgeForwardMsg(BaseModel):
    executor_origin: ExecutorOrigin
    msg: ExecuteProgramMsgForFundsForDisplayedForUint128
    to: NetworkId = Field(..., description='target network')


class ConfigSubMsg7(BaseModel):
    """
    short cut to rollout config faster
    """

    class Config:
        extra = Extra.forbid

    force: List[ConfigSubMsg]


class ConfigSubMsg(BaseModel):
    __root__: Union[
        ConfigSubMsg1,
        ConfigSubMsg2,
        ConfigSubMsg3,
        ConfigSubMsg4,
        ConfigSubMsg5,
        ConfigSubMsg6,
        ConfigSubMsg7,
        ConfigSubMsg8,
    ] = Field(
        ...,
        description='cross cross chain routing requires a lot of configuration, about chain executing this contract, about connectivity to and of other chains (even if not connected directly) and about assets and services on these chains (in future block hooks and some set of host extensions/precompiles would help to get some info automatically) `Force` message sets the data unconditionally.',
    )


class ExecuteProgramMsgForFundsForDisplayedForUint128(BaseModel):
    """
    Definition of a program to be executed including its context.
    """

    assets: FundsForDisplayedForUint128 = Field(
        ...,
        description='Assets to fund the CVM interpreter instance. The interpreter is funded prior to execution. If None, 100% of received funds go to interpreter.',
    )
    program: ProgramForArrayOfInstructionForArrayOfUint8AndXcAddrAndFundsForAmount = (
        Field(..., description='The program.')
    )
    salt: Optional[str] = Field(
        None,
        description='The program salt. If JSON, than hex encoded non prefixed lower case string. If not specified, uses no salt.',
    )
    tip: Optional[str] = None


class ExecuteProgramMsgForNullableFundsForDisplayedForUint128(BaseModel):
    """
    Definition of a program to be executed including its context.
    """

    assets: Optional[FundsForDisplayedForUint128] = Field(
        None,
        description='Assets to fund the CVM interpreter instance. The interpreter is funded prior to execution. If None, 100% of received funds go to interpreter.',
    )
    program: ProgramForArrayOfInstructionForArrayOfUint8AndXcAddrAndFundsForAmount = (
        Field(..., description='The program.')
    )
    salt: Optional[str] = Field(
        None,
        description='The program salt. If JSON, than hex encoded non prefixed lower case string. If not specified, uses no salt.',
    )
    tip: Optional[str] = None


class Spawn(BaseModel):
    assets: FundsForAmount
    network_id: NetworkId
    program: ProgramForArrayOfInstructionForArrayOfUint8AndXcAddrAndFundsForAmount
    salt: Optional[str] = Field(
        None,
        description='If JSON, than hex encoded non prefixed lower case string. Different salt allows to split funds into different virtual wallets So same salt shares assets on set of derived accounts on chains program executes.',
    )


class InstructionForArrayOfUint8AndXcAddrAndFundsForAmount3(BaseModel):
    """
    Spawn a sub-program on the target `network`.

    The program will be spawned with the desired [`Assets`]. The salt is used to track the program when events are dispatched in the network.
    """

    class Config:
        extra = Extra.forbid

    spawn: Spawn


class InstructionForArrayOfUint8AndXcAddrAndFundsForAmount(BaseModel):
    __root__: Union[
        InstructionForArrayOfUint8AndXcAddrAndFundsForAmount1,
        InstructionForArrayOfUint8AndXcAddrAndFundsForAmount2,
        InstructionForArrayOfUint8AndXcAddrAndFundsForAmount3,
        InstructionForArrayOfUint8AndXcAddrAndFundsForAmount4,
    ] = Field(
        ...,
        description='Base XCVM instructions. This set will remain as small as possible, expressiveness must come on `top` of the base instructions.',
    )


class PacketForProgramForArrayOfInstructionForArrayOfUint8AndXcAddrAndFundsForAmount(
    BaseModel
):
    assets: FundsForDisplayedForUint128 = Field(
        ..., description='The assets that were attached to the program.'
    )
    interpreter: str = Field(
        ..., description='The interpreter that was the origin of this packet.'
    )
    program: ProgramForArrayOfInstructionForArrayOfUint8AndXcAddrAndFundsForAmount = (
        Field(..., description='The protobuf encoded program.')
    )
    salt: str = Field(..., description='The salt associated with the program.')
    user_origin: UserOrigin = Field(
        ..., description='The user that originated the first XCVM call.'
    )


class ProgramForArrayOfInstructionForArrayOfUint8AndXcAddrAndFundsForAmount(BaseModel):
    instructions: List[InstructionForArrayOfUint8AndXcAddrAndFundsForAmount] = Field(
        ..., description='list of instructions to be executed'
    )
    tag: Optional[str] = Field(
        None,
        description='In JSON, hex encoded identifiers to identify the program off chain (for example in indexer).',
    )


class XcMessageData(BaseModel):
    """
    This message should be send as part of wasm termination memo. So that can match it to sender hash and know what channel and origin was used to send message. All information here is not secured until compared with existing secured data.
    """

    from_network_id: NetworkId
    packet: PacketForProgramForArrayOfInstructionForArrayOfUint8AndXcAddrAndFundsForAmount


ExecuteMsg1.update_forward_refs()
ExecuteMsg2.update_forward_refs()
ExecuteProgramPrivileged.update_forward_refs()
ExecuteMsg4.update_forward_refs()
ExecuteMsg6.update_forward_refs()
BridgeForwardMsg.update_forward_refs()
ConfigSubMsg7.update_forward_refs()
ExecuteProgramMsgForFundsForDisplayedForUint128.update_forward_refs()
ExecuteProgramMsgForNullableFundsForDisplayedForUint128.update_forward_refs()
Spawn.update_forward_refs()
PacketForProgramForArrayOfInstructionForArrayOfUint8AndXcAddrAndFundsForAmount.update_forward_refs()
