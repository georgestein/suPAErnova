??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02unknown8??
|
dense_428/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_428/kernel
u
$dense_428/kernel/Read/ReadVariableOpReadVariableOpdense_428/kernel*
_output_shapes

: *
dtype0
t
dense_428/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_428/bias
m
"dense_428/bias/Read/ReadVariableOpReadVariableOpdense_428/bias*
_output_shapes
: *
dtype0
?
color_law_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_namecolor_law_56/kernel
|
'color_law_56/kernel/Read/ReadVariableOpReadVariableOpcolor_law_56/kernel*
_output_shapes
:	?*
dtype0
}
dense_429/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*!
shared_namedense_429/kernel
v
$dense_429/kernel/Read/ReadVariableOpReadVariableOpdense_429/kernel*
_output_shapes
:	 ?*
dtype0
u
dense_429/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_429/bias
n
"dense_429/bias/Read/ReadVariableOpReadVariableOpdense_429/bias*
_output_shapes	
:?*
dtype0
~
dense_430/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_430/kernel
w
$dense_430/kernel/Read/ReadVariableOpReadVariableOpdense_430/kernel* 
_output_shapes
:
??*
dtype0
u
dense_430/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_430/bias
n
"dense_430/bias/Read/ReadVariableOpReadVariableOpdense_430/bias*
_output_shapes	
:?*
dtype0
~
dense_431/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_431/kernel
w
$dense_431/kernel/Read/ReadVariableOpReadVariableOpdense_431/kernel* 
_output_shapes
:
??*
dtype0
u
dense_431/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_431/bias
n
"dense_431/bias/Read/ReadVariableOpReadVariableOpdense_431/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?9B?9 B?9
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-0
	layer-8

layer_with_weights-1

layer-9
layer-10
layer_with_weights-2
layer-11
layer-12
layer_with_weights-3
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
 
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
R
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
^

:kernel
;	variables
<regularization_losses
=trainable_variables
>	keras_api
R
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
h

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
R
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
h

Mkernel
Nbias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
R
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
h

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
R
]	variables
^regularization_losses
_trainable_variables
`	keras_api
R
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
 
R
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
R
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
R
m	variables
nregularization_losses
otrainable_variables
p	keras_api
?
40
51
:2
C3
D4
M5
N6
W7
X8
 
8
40
51
C2
D3
M4
N5
W6
X7
?
	variables
qlayer_regularization_losses
rnon_trainable_variables
smetrics
tlayer_metrics
regularization_losses

ulayers
trainable_variables
 
 
 
 
?
	variables
vlayer_regularization_losses
wnon_trainable_variables
xmetrics
ylayer_metrics
regularization_losses

zlayers
trainable_variables
 
 
 
?
 	variables
{layer_regularization_losses
|non_trainable_variables
}metrics
~layer_metrics
!regularization_losses

layers
"trainable_variables
 
 
 
?
$	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
%regularization_losses
?layers
&trainable_variables
 
 
 
?
(	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
)regularization_losses
?layers
*trainable_variables
 
 
 
?
,	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
-regularization_losses
?layers
.trainable_variables
 
 
 
?
0	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
1regularization_losses
?layers
2trainable_variables
\Z
VARIABLE_VALUEdense_428/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_428/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
?
6	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
7regularization_losses
?layers
8trainable_variables
_]
VARIABLE_VALUEcolor_law_56/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE

:0
 
 
?
;	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
<regularization_losses
?layers
=trainable_variables
 
 
 
?
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
@regularization_losses
?layers
Atrainable_variables
\Z
VARIABLE_VALUEdense_429/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_429/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
?
E	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Fregularization_losses
?layers
Gtrainable_variables
 
 
 
?
I	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Jregularization_losses
?layers
Ktrainable_variables
\Z
VARIABLE_VALUEdense_430/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_430/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
 

M0
N1
?
O	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Pregularization_losses
?layers
Qtrainable_variables
 
 
 
?
S	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Tregularization_losses
?layers
Utrainable_variables
\Z
VARIABLE_VALUEdense_431/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_431/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
?
Y	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Zregularization_losses
?layers
[trainable_variables
 
 
 
?
]	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
^regularization_losses
?layers
_trainable_variables
 
 
 
?
a	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
bregularization_losses
?layers
ctrainable_variables
 
 
 
?
e	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
fregularization_losses
?layers
gtrainable_variables
 
 
 
?
i	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
jregularization_losses
?layers
ktrainable_variables
 
 
 
?
m	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
nregularization_losses
?layers
otrainable_variables
 

:0
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

:0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
"serving_default_conditional_paramsPlaceholder*+
_output_shapes
:????????? *
dtype0* 
shape:????????? 
?
serving_default_input_216Placeholder*,
_output_shapes
:????????? ?*
dtype0*!
shape:????????? ?
?
serving_default_latent_paramsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_conditional_paramsserving_default_input_216serving_default_latent_paramscolor_law_56/kerneldense_428/kerneldense_428/biasdense_429/kerneldense_429/biasdense_430/kerneldense_430/biasdense_431/kerneldense_431/bias*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_446577
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_428/kernel/Read/ReadVariableOp"dense_428/bias/Read/ReadVariableOp'color_law_56/kernel/Read/ReadVariableOp$dense_429/kernel/Read/ReadVariableOp"dense_429/bias/Read/ReadVariableOp$dense_430/kernel/Read/ReadVariableOp"dense_430/bias/Read/ReadVariableOp$dense_431/kernel/Read/ReadVariableOp"dense_431/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_447370
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_428/kerneldense_428/biascolor_law_56/kerneldense_429/kerneldense_429/biasdense_430/kerneldense_430/biasdense_431/kerneldense_431/bias*
Tin
2
*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_447409??
?
?
E__inference_dense_428_layer_call_and_return_conditional_losses_446113

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2	
BiasAddW
ReluReluBiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? :::S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_model_107_layer_call_fn_446965
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_model_107_layer_call_and_return_conditional_losses_4465292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
*__inference_model_107_layer_call_fn_446482
latent_params
conditional_params
	input_216
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_216unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_model_107_layer_call_and_return_conditional_losses_4464612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namelatent_params:_[
+
_output_shapes
:????????? 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:????????? ?
#
_user_specified_name	input_216:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?(
?
__inference__traced_save_447370
file_prefix/
+savev2_dense_428_kernel_read_readvariableop-
)savev2_dense_428_bias_read_readvariableop2
.savev2_color_law_56_kernel_read_readvariableop/
+savev2_dense_429_kernel_read_readvariableop-
)savev2_dense_429_bias_read_readvariableop/
+savev2_dense_430_kernel_read_readvariableop-
)savev2_dense_430_bias_read_readvariableop/
+savev2_dense_431_kernel_read_readvariableop-
)savev2_dense_431_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_49106fcf77b9400d937f0c0ba64d4c4c/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_428_kernel_read_readvariableop)savev2_dense_428_bias_read_readvariableop.savev2_color_law_56_kernel_read_readvariableop+savev2_dense_429_kernel_read_readvariableop)savev2_dense_429_bias_read_readvariableop+savev2_dense_430_kernel_read_readvariableop)savev2_dense_430_bias_read_readvariableop+savev2_dense_431_kernel_read_readvariableop)savev2_dense_431_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*j
_input_shapesY
W: : : :	?:	 ?:?:
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :%!

_output_shapes
:	?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!	

_output_shapes	
:?:


_output_shapes
: 
?
u
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_447024

inputs
identity?
strided_slice_434/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_434/begin
strided_slice_434/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_434/end?
strided_slice_434/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_434/strides?
strided_slice_434StridedSliceinputs strided_slice_434/begin:output:0strided_slice_434/end:output:0"strided_slice_434/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_434r
IdentityIdentitystrided_slice_434:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
6__inference_tf_op_layer_AddV2_107_layer_call_fn_447168
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_4461352
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:????????? ?:????????? :V R
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
P
4__inference_tf_op_layer_Mul_328_layer_call_fn_447219

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_4461972
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?

*__inference_dense_428_layer_call_fn_447069

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_428_layer_call_and_return_conditional_losses_4461132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
`
4__inference_tf_op_layer_Mul_330_layer_call_fn_447314
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4463602
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:????????? ?:????????? :V R
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
Z
>__inference_tf_op_layer_strided_slice_432_layer_call_fn_446978

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_4459562
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_model_107_layer_call_fn_446940
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_model_107_layer_call_and_return_conditional_losses_4464612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
}
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_446997
inputs_0
inputs_1
identityx
	AddV2_106AddV2inputs_0inputs_1*
T0*
_cloned(*+
_output_shapes
:????????? 2
	AddV2_106e
IdentityIdentityAddV2_106:z:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????? :????????? :U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
j
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_446304

inputs
identityY
Pow_53/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

Pow_53/xx
Pow_53PowPow_53/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2
Pow_53c
IdentityIdentity
Pow_53:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?.
?
"__inference__traced_restore_447409
file_prefix%
!assignvariableop_dense_428_kernel%
!assignvariableop_1_dense_428_bias*
&assignvariableop_2_color_law_56_kernel'
#assignvariableop_3_dense_429_kernel%
!assignvariableop_4_dense_429_bias'
#assignvariableop_5_dense_430_kernel%
!assignvariableop_6_dense_430_bias'
#assignvariableop_7_dense_431_kernel%
!assignvariableop_8_dense_431_bias
identity_10??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_428_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_428_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_color_law_56_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_429_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_429_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_430_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_430_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_431_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_431_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_9?
Identity_10IdentityIdentity_9:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_10"#
identity_10Identity_10:output:0*9
_input_shapes(
&: :::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?K
?
E__inference_model_107_layer_call_and_return_conditional_losses_446413
latent_params
conditional_params
	input_216
color_law_446381
dense_428_446385
dense_428_446387
dense_429_446391
dense_429_446393
dense_430_446397
dense_430_446399
dense_431_446402
dense_431_446404
identity??!color_law/StatefulPartitionedCall?!dense_428/StatefulPartitionedCall?!dense_429/StatefulPartitionedCall?!dense_430/StatefulPartitionedCall?!dense_431/StatefulPartitionedCall?
 repeat_vector_53/PartitionedCallPartitionedCalllatent_params*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_repeat_vector_53_layer_call_and_return_conditional_losses_4459352"
 repeat_vector_53/PartitionedCall?
-tf_op_layer_strided_slice_432/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_4459562/
-tf_op_layer_strided_slice_432/PartitionedCall?
-tf_op_layer_strided_slice_435/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_4459722/
-tf_op_layer_strided_slice_435/PartitionedCall?
%tf_op_layer_AddV2_106/PartitionedCallPartitionedCallconditional_params6tf_op_layer_strided_slice_432/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_4459862'
%tf_op_layer_AddV2_106/PartitionedCall?
-tf_op_layer_strided_slice_434/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_4460032/
-tf_op_layer_strided_slice_434/PartitionedCall?
concatenate_161/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_435/PartitionedCall:output:0.tf_op_layer_AddV2_106/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_161_layer_call_and_return_conditional_losses_4460182!
concatenate_161/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_434/PartitionedCall:output:0color_law_446381*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4460542#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_433/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_4460742/
-tf_op_layer_strided_slice_433/PartitionedCall?
!dense_428/StatefulPartitionedCallStatefulPartitionedCall(concatenate_161/PartitionedCall:output:0dense_428_446385dense_428_446387*
Tin
2*
Tout
2*+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_428_layer_call_and_return_conditional_losses_4461132#
!dense_428/StatefulPartitionedCall?
%tf_op_layer_AddV2_107/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_433/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_4461352'
%tf_op_layer_AddV2_107/PartitionedCall?
!dense_429/StatefulPartitionedCallStatefulPartitionedCall*dense_428/StatefulPartitionedCall:output:0dense_429_446391dense_429_446393*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_429_layer_call_and_return_conditional_losses_4461752#
!dense_429/StatefulPartitionedCall?
#tf_op_layer_Mul_328/PartitionedCallPartitionedCall.tf_op_layer_AddV2_107/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_4461972%
#tf_op_layer_Mul_328/PartitionedCall?
!dense_430/StatefulPartitionedCallStatefulPartitionedCall*dense_429/StatefulPartitionedCall:output:0dense_430_446397dense_430_446399*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_430_layer_call_and_return_conditional_losses_4462362#
!dense_430/StatefulPartitionedCall?
!dense_431/StatefulPartitionedCallStatefulPartitionedCall*dense_430/StatefulPartitionedCall:output:0dense_431_446402dense_431_446404*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_431_layer_call_and_return_conditional_losses_4462822#
!dense_431/StatefulPartitionedCall?
"tf_op_layer_Pow_53/PartitionedCallPartitionedCall,tf_op_layer_Mul_328/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_4463042$
"tf_op_layer_Pow_53/PartitionedCall?
#tf_op_layer_Mul_329/PartitionedCallPartitionedCall*dense_431/StatefulPartitionedCall:output:0+tf_op_layer_Pow_53/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_4463182%
#tf_op_layer_Mul_329/PartitionedCall?
#tf_op_layer_Relu_49/PartitionedCallPartitionedCall,tf_op_layer_Mul_329/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_4463322%
#tf_op_layer_Relu_49/PartitionedCall?
"tf_op_layer_Max_57/PartitionedCallPartitionedCall	input_216*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_4463462$
"tf_op_layer_Max_57/PartitionedCall?
#tf_op_layer_Mul_330/PartitionedCallPartitionedCall,tf_op_layer_Relu_49/PartitionedCall:output:0+tf_op_layer_Max_57/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4463602%
#tf_op_layer_Mul_330/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_330/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dense_429/StatefulPartitionedCall"^dense_430/StatefulPartitionedCall"^dense_431/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dense_429/StatefulPartitionedCall!dense_429/StatefulPartitionedCall2F
!dense_430/StatefulPartitionedCall!dense_430/StatefulPartitionedCall2F
!dense_431/StatefulPartitionedCall!dense_431/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namelatent_params:_[
+
_output_shapes
:????????? 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:????????? ?
#
_user_specified_name	input_216:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
u
K__inference_concatenate_161_layer_call_and_return_conditional_losses_446018

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????? :????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:SO
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
u
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_446074

inputs
identity?
strided_slice_433/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_433/begin
strided_slice_433/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_433/end?
strided_slice_433/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_433/strides?
strided_slice_433StridedSliceinputs strided_slice_433/begin:output:0strided_slice_433/end:output:0"strided_slice_433/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_433r
IdentityIdentitystrided_slice_433:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
O
3__inference_tf_op_layer_Max_57_layer_call_fn_447302

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_4463462
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
{
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_446135

inputs
inputs_1
identityw
	AddV2_107AddV2inputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2
	AddV2_107f
IdentityIdentityAddV2_107:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:????????? ?:????????? :T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:SO
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_color_law_layer_call_and_return_conditional_losses_447096

inputs%
!tensordot_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
	Tensordotk
IdentityIdentityTensordot:output:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: 
?
M
1__inference_repeat_vector_53_layer_call_fn_445941

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*4
_output_shapes"
 :????????? ?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_repeat_vector_53_layer_call_and_return_conditional_losses_4459352
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :????????? ?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?K
?
E__inference_model_107_layer_call_and_return_conditional_losses_446461

inputs
inputs_1
inputs_2
color_law_446429
dense_428_446433
dense_428_446435
dense_429_446439
dense_429_446441
dense_430_446445
dense_430_446447
dense_431_446450
dense_431_446452
identity??!color_law/StatefulPartitionedCall?!dense_428/StatefulPartitionedCall?!dense_429/StatefulPartitionedCall?!dense_430/StatefulPartitionedCall?!dense_431/StatefulPartitionedCall?
 repeat_vector_53/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_repeat_vector_53_layer_call_and_return_conditional_losses_4459352"
 repeat_vector_53/PartitionedCall?
-tf_op_layer_strided_slice_432/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_4459562/
-tf_op_layer_strided_slice_432/PartitionedCall?
-tf_op_layer_strided_slice_435/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_4459722/
-tf_op_layer_strided_slice_435/PartitionedCall?
%tf_op_layer_AddV2_106/PartitionedCallPartitionedCallinputs_16tf_op_layer_strided_slice_432/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_4459862'
%tf_op_layer_AddV2_106/PartitionedCall?
-tf_op_layer_strided_slice_434/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_4460032/
-tf_op_layer_strided_slice_434/PartitionedCall?
concatenate_161/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_435/PartitionedCall:output:0.tf_op_layer_AddV2_106/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_161_layer_call_and_return_conditional_losses_4460182!
concatenate_161/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_434/PartitionedCall:output:0color_law_446429*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4460542#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_433/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_4460742/
-tf_op_layer_strided_slice_433/PartitionedCall?
!dense_428/StatefulPartitionedCallStatefulPartitionedCall(concatenate_161/PartitionedCall:output:0dense_428_446433dense_428_446435*
Tin
2*
Tout
2*+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_428_layer_call_and_return_conditional_losses_4461132#
!dense_428/StatefulPartitionedCall?
%tf_op_layer_AddV2_107/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_433/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_4461352'
%tf_op_layer_AddV2_107/PartitionedCall?
!dense_429/StatefulPartitionedCallStatefulPartitionedCall*dense_428/StatefulPartitionedCall:output:0dense_429_446439dense_429_446441*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_429_layer_call_and_return_conditional_losses_4461752#
!dense_429/StatefulPartitionedCall?
#tf_op_layer_Mul_328/PartitionedCallPartitionedCall.tf_op_layer_AddV2_107/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_4461972%
#tf_op_layer_Mul_328/PartitionedCall?
!dense_430/StatefulPartitionedCallStatefulPartitionedCall*dense_429/StatefulPartitionedCall:output:0dense_430_446445dense_430_446447*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_430_layer_call_and_return_conditional_losses_4462362#
!dense_430/StatefulPartitionedCall?
!dense_431/StatefulPartitionedCallStatefulPartitionedCall*dense_430/StatefulPartitionedCall:output:0dense_431_446450dense_431_446452*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_431_layer_call_and_return_conditional_losses_4462822#
!dense_431/StatefulPartitionedCall?
"tf_op_layer_Pow_53/PartitionedCallPartitionedCall,tf_op_layer_Mul_328/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_4463042$
"tf_op_layer_Pow_53/PartitionedCall?
#tf_op_layer_Mul_329/PartitionedCallPartitionedCall*dense_431/StatefulPartitionedCall:output:0+tf_op_layer_Pow_53/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_4463182%
#tf_op_layer_Mul_329/PartitionedCall?
#tf_op_layer_Relu_49/PartitionedCallPartitionedCall,tf_op_layer_Mul_329/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_4463322%
#tf_op_layer_Relu_49/PartitionedCall?
"tf_op_layer_Max_57/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_4463462$
"tf_op_layer_Max_57/PartitionedCall?
#tf_op_layer_Mul_330/PartitionedCallPartitionedCall,tf_op_layer_Relu_49/PartitionedCall:output:0+tf_op_layer_Max_57/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4463602%
#tf_op_layer_Mul_330/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_330/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dense_429/StatefulPartitionedCall"^dense_430/StatefulPartitionedCall"^dense_431/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dense_429/StatefulPartitionedCall!dense_429/StatefulPartitionedCall2F
!dense_430/StatefulPartitionedCall!dense_430/StatefulPartitionedCall2F
!dense_431/StatefulPartitionedCall!dense_431/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:TP
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
Z
>__inference_tf_op_layer_strided_slice_433_layer_call_fn_447116

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_4460742
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?K
?
E__inference_model_107_layer_call_and_return_conditional_losses_446529

inputs
inputs_1
inputs_2
color_law_446497
dense_428_446501
dense_428_446503
dense_429_446507
dense_429_446509
dense_430_446513
dense_430_446515
dense_431_446518
dense_431_446520
identity??!color_law/StatefulPartitionedCall?!dense_428/StatefulPartitionedCall?!dense_429/StatefulPartitionedCall?!dense_430/StatefulPartitionedCall?!dense_431/StatefulPartitionedCall?
 repeat_vector_53/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_repeat_vector_53_layer_call_and_return_conditional_losses_4459352"
 repeat_vector_53/PartitionedCall?
-tf_op_layer_strided_slice_432/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_4459562/
-tf_op_layer_strided_slice_432/PartitionedCall?
-tf_op_layer_strided_slice_435/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_4459722/
-tf_op_layer_strided_slice_435/PartitionedCall?
%tf_op_layer_AddV2_106/PartitionedCallPartitionedCallinputs_16tf_op_layer_strided_slice_432/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_4459862'
%tf_op_layer_AddV2_106/PartitionedCall?
-tf_op_layer_strided_slice_434/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_4460032/
-tf_op_layer_strided_slice_434/PartitionedCall?
concatenate_161/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_435/PartitionedCall:output:0.tf_op_layer_AddV2_106/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_161_layer_call_and_return_conditional_losses_4460182!
concatenate_161/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_434/PartitionedCall:output:0color_law_446497*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4460542#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_433/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_4460742/
-tf_op_layer_strided_slice_433/PartitionedCall?
!dense_428/StatefulPartitionedCallStatefulPartitionedCall(concatenate_161/PartitionedCall:output:0dense_428_446501dense_428_446503*
Tin
2*
Tout
2*+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_428_layer_call_and_return_conditional_losses_4461132#
!dense_428/StatefulPartitionedCall?
%tf_op_layer_AddV2_107/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_433/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_4461352'
%tf_op_layer_AddV2_107/PartitionedCall?
!dense_429/StatefulPartitionedCallStatefulPartitionedCall*dense_428/StatefulPartitionedCall:output:0dense_429_446507dense_429_446509*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_429_layer_call_and_return_conditional_losses_4461752#
!dense_429/StatefulPartitionedCall?
#tf_op_layer_Mul_328/PartitionedCallPartitionedCall.tf_op_layer_AddV2_107/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_4461972%
#tf_op_layer_Mul_328/PartitionedCall?
!dense_430/StatefulPartitionedCallStatefulPartitionedCall*dense_429/StatefulPartitionedCall:output:0dense_430_446513dense_430_446515*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_430_layer_call_and_return_conditional_losses_4462362#
!dense_430/StatefulPartitionedCall?
!dense_431/StatefulPartitionedCallStatefulPartitionedCall*dense_430/StatefulPartitionedCall:output:0dense_431_446518dense_431_446520*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_431_layer_call_and_return_conditional_losses_4462822#
!dense_431/StatefulPartitionedCall?
"tf_op_layer_Pow_53/PartitionedCallPartitionedCall,tf_op_layer_Mul_328/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_4463042$
"tf_op_layer_Pow_53/PartitionedCall?
#tf_op_layer_Mul_329/PartitionedCallPartitionedCall*dense_431/StatefulPartitionedCall:output:0+tf_op_layer_Pow_53/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_4463182%
#tf_op_layer_Mul_329/PartitionedCall?
#tf_op_layer_Relu_49/PartitionedCallPartitionedCall,tf_op_layer_Mul_329/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_4463322%
#tf_op_layer_Relu_49/PartitionedCall?
"tf_op_layer_Max_57/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_4463462$
"tf_op_layer_Max_57/PartitionedCall?
#tf_op_layer_Mul_330/PartitionedCallPartitionedCall,tf_op_layer_Relu_49/PartitionedCall:output:0+tf_op_layer_Max_57/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4463602%
#tf_op_layer_Mul_330/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_330/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dense_429/StatefulPartitionedCall"^dense_430/StatefulPartitionedCall"^dense_431/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dense_429/StatefulPartitionedCall!dense_429/StatefulPartitionedCall2F
!dense_430/StatefulPartitionedCall!dense_430/StatefulPartitionedCall2F
!dense_431/StatefulPartitionedCall!dense_431/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:TP
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
P
4__inference_tf_op_layer_Relu_49_layer_call_fn_447291

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_4463322
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
k
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_446332

inputs
identityh
Relu_49Reluinputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Relu_49n
IdentityIdentityRelu_49:activations:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
u
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_446003

inputs
identity?
strided_slice_434/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_434/begin
strided_slice_434/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_434/end?
strided_slice_434/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_434/strides?
strided_slice_434StridedSliceinputs strided_slice_434/begin:output:0strided_slice_434/end:output:0"strided_slice_434/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_434r
IdentityIdentitystrided_slice_434:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
Z
>__inference_tf_op_layer_strided_slice_434_layer_call_fn_447029

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_4460032
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_dense_429_layer_call_and_return_conditional_losses_447147

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2	
BiasAddX
ReluReluBiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????  :::S O
+
_output_shapes
:?????????  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
k
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_447286

inputs
identityh
Relu_49Reluinputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Relu_49n
IdentityIdentityRelu_49:activations:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
u
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_447111

inputs
identity?
strided_slice_433/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_433/begin
strided_slice_433/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_433/end?
strided_slice_433/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_433/strides?
strided_slice_433StridedSliceinputs strided_slice_433/begin:output:0strided_slice_433/end:output:0"strided_slice_433/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_433r
IdentityIdentitystrided_slice_433:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_dense_430_layer_call_and_return_conditional_losses_447199

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2	
BiasAddX
ReluReluBiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :????????? ?:::T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_dense_430_layer_call_and_return_conditional_losses_446236

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2	
BiasAddX
ReluReluBiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :????????? ?:::T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_447297

inputs
identity
Max_57/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max_57/reduction_indices?
Max_57Maxinputs!Max_57/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
Max_57g
IdentityIdentityMax_57:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
j
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_446346

inputs
identity
Max_57/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max_57/reduction_indices?
Max_57Maxinputs!Max_57/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
Max_57g
IdentityIdentityMax_57:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_445926
latent_params
conditional_params
	input_2169
5model_107_color_law_tensordot_readvariableop_resource9
5model_107_dense_428_tensordot_readvariableop_resource7
3model_107_dense_428_biasadd_readvariableop_resource9
5model_107_dense_429_tensordot_readvariableop_resource7
3model_107_dense_429_biasadd_readvariableop_resource9
5model_107_dense_430_tensordot_readvariableop_resource7
3model_107_dense_430_biasadd_readvariableop_resource9
5model_107_dense_431_tensordot_readvariableop_resource7
3model_107_dense_431_biasadd_readvariableop_resource
identity??
)model_107/repeat_vector_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_107/repeat_vector_53/ExpandDims/dim?
%model_107/repeat_vector_53/ExpandDims
ExpandDimslatent_params2model_107/repeat_vector_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_107/repeat_vector_53/ExpandDims?
 model_107/repeat_vector_53/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_107/repeat_vector_53/stack?
model_107/repeat_vector_53/TileTile.model_107/repeat_vector_53/ExpandDims:output:0)model_107/repeat_vector_53/stack:output:0*
T0*+
_output_shapes
:????????? 2!
model_107/repeat_vector_53/Tile?
?model_107/tf_op_layer_strided_slice_432/strided_slice_432/beginConst*
_output_shapes
:*
dtype0*
valueB"        2A
?model_107/tf_op_layer_strided_slice_432/strided_slice_432/begin?
=model_107/tf_op_layer_strided_slice_432/strided_slice_432/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_107/tf_op_layer_strided_slice_432/strided_slice_432/end?
Amodel_107/tf_op_layer_strided_slice_432/strided_slice_432/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_107/tf_op_layer_strided_slice_432/strided_slice_432/strides?
9model_107/tf_op_layer_strided_slice_432/strided_slice_432StridedSlice(model_107/repeat_vector_53/Tile:output:0Hmodel_107/tf_op_layer_strided_slice_432/strided_slice_432/begin:output:0Fmodel_107/tf_op_layer_strided_slice_432/strided_slice_432/end:output:0Jmodel_107/tf_op_layer_strided_slice_432/strided_slice_432/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_107/tf_op_layer_strided_slice_432/strided_slice_432?
?model_107/tf_op_layer_strided_slice_435/strided_slice_435/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_107/tf_op_layer_strided_slice_435/strided_slice_435/begin?
=model_107/tf_op_layer_strided_slice_435/strided_slice_435/endConst*
_output_shapes
:*
dtype0*
valueB"        2?
=model_107/tf_op_layer_strided_slice_435/strided_slice_435/end?
Amodel_107/tf_op_layer_strided_slice_435/strided_slice_435/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_107/tf_op_layer_strided_slice_435/strided_slice_435/strides?
9model_107/tf_op_layer_strided_slice_435/strided_slice_435StridedSlice(model_107/repeat_vector_53/Tile:output:0Hmodel_107/tf_op_layer_strided_slice_435/strided_slice_435/begin:output:0Fmodel_107/tf_op_layer_strided_slice_435/strided_slice_435/end:output:0Jmodel_107/tf_op_layer_strided_slice_435/strided_slice_435/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2;
9model_107/tf_op_layer_strided_slice_435/strided_slice_435?
)model_107/tf_op_layer_AddV2_106/AddV2_106AddV2conditional_paramsBmodel_107/tf_op_layer_strided_slice_432/strided_slice_432:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2+
)model_107/tf_op_layer_AddV2_106/AddV2_106?
?model_107/tf_op_layer_strided_slice_434/strided_slice_434/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_107/tf_op_layer_strided_slice_434/strided_slice_434/begin?
=model_107/tf_op_layer_strided_slice_434/strided_slice_434/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_107/tf_op_layer_strided_slice_434/strided_slice_434/end?
Amodel_107/tf_op_layer_strided_slice_434/strided_slice_434/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_107/tf_op_layer_strided_slice_434/strided_slice_434/strides?
9model_107/tf_op_layer_strided_slice_434/strided_slice_434StridedSlice(model_107/repeat_vector_53/Tile:output:0Hmodel_107/tf_op_layer_strided_slice_434/strided_slice_434/begin:output:0Fmodel_107/tf_op_layer_strided_slice_434/strided_slice_434/end:output:0Jmodel_107/tf_op_layer_strided_slice_434/strided_slice_434/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_107/tf_op_layer_strided_slice_434/strided_slice_434?
%model_107/concatenate_161/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_107/concatenate_161/concat/axis?
 model_107/concatenate_161/concatConcatV2Bmodel_107/tf_op_layer_strided_slice_435/strided_slice_435:output:0-model_107/tf_op_layer_AddV2_106/AddV2_106:z:0.model_107/concatenate_161/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2"
 model_107/concatenate_161/concat?
,model_107/color_law/Tensordot/ReadVariableOpReadVariableOp5model_107_color_law_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,model_107/color_law/Tensordot/ReadVariableOp?
"model_107/color_law/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_107/color_law/Tensordot/axes?
"model_107/color_law/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_107/color_law/Tensordot/free?
#model_107/color_law/Tensordot/ShapeShapeBmodel_107/tf_op_layer_strided_slice_434/strided_slice_434:output:0*
T0*
_output_shapes
:2%
#model_107/color_law/Tensordot/Shape?
+model_107/color_law/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/color_law/Tensordot/GatherV2/axis?
&model_107/color_law/Tensordot/GatherV2GatherV2,model_107/color_law/Tensordot/Shape:output:0+model_107/color_law/Tensordot/free:output:04model_107/color_law/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_107/color_law/Tensordot/GatherV2?
-model_107/color_law/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_107/color_law/Tensordot/GatherV2_1/axis?
(model_107/color_law/Tensordot/GatherV2_1GatherV2,model_107/color_law/Tensordot/Shape:output:0+model_107/color_law/Tensordot/axes:output:06model_107/color_law/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_107/color_law/Tensordot/GatherV2_1?
#model_107/color_law/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_107/color_law/Tensordot/Const?
"model_107/color_law/Tensordot/ProdProd/model_107/color_law/Tensordot/GatherV2:output:0,model_107/color_law/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_107/color_law/Tensordot/Prod?
%model_107/color_law/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_107/color_law/Tensordot/Const_1?
$model_107/color_law/Tensordot/Prod_1Prod1model_107/color_law/Tensordot/GatherV2_1:output:0.model_107/color_law/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_107/color_law/Tensordot/Prod_1?
)model_107/color_law/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_107/color_law/Tensordot/concat/axis?
$model_107/color_law/Tensordot/concatConcatV2+model_107/color_law/Tensordot/free:output:0+model_107/color_law/Tensordot/axes:output:02model_107/color_law/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_107/color_law/Tensordot/concat?
#model_107/color_law/Tensordot/stackPack+model_107/color_law/Tensordot/Prod:output:0-model_107/color_law/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_107/color_law/Tensordot/stack?
'model_107/color_law/Tensordot/transpose	TransposeBmodel_107/tf_op_layer_strided_slice_434/strided_slice_434:output:0-model_107/color_law/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2)
'model_107/color_law/Tensordot/transpose?
%model_107/color_law/Tensordot/ReshapeReshape+model_107/color_law/Tensordot/transpose:y:0,model_107/color_law/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_107/color_law/Tensordot/Reshape?
$model_107/color_law/Tensordot/MatMulMatMul.model_107/color_law/Tensordot/Reshape:output:04model_107/color_law/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_107/color_law/Tensordot/MatMul?
%model_107/color_law/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_107/color_law/Tensordot/Const_2?
+model_107/color_law/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/color_law/Tensordot/concat_1/axis?
&model_107/color_law/Tensordot/concat_1ConcatV2/model_107/color_law/Tensordot/GatherV2:output:0.model_107/color_law/Tensordot/Const_2:output:04model_107/color_law/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_107/color_law/Tensordot/concat_1?
model_107/color_law/TensordotReshape.model_107/color_law/Tensordot/MatMul:product:0/model_107/color_law/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_107/color_law/Tensordot?
?model_107/tf_op_layer_strided_slice_433/strided_slice_433/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_107/tf_op_layer_strided_slice_433/strided_slice_433/begin?
=model_107/tf_op_layer_strided_slice_433/strided_slice_433/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_107/tf_op_layer_strided_slice_433/strided_slice_433/end?
Amodel_107/tf_op_layer_strided_slice_433/strided_slice_433/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_107/tf_op_layer_strided_slice_433/strided_slice_433/strides?
9model_107/tf_op_layer_strided_slice_433/strided_slice_433StridedSlice(model_107/repeat_vector_53/Tile:output:0Hmodel_107/tf_op_layer_strided_slice_433/strided_slice_433/begin:output:0Fmodel_107/tf_op_layer_strided_slice_433/strided_slice_433/end:output:0Jmodel_107/tf_op_layer_strided_slice_433/strided_slice_433/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_107/tf_op_layer_strided_slice_433/strided_slice_433?
,model_107/dense_428/Tensordot/ReadVariableOpReadVariableOp5model_107_dense_428_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02.
,model_107/dense_428/Tensordot/ReadVariableOp?
"model_107/dense_428/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_107/dense_428/Tensordot/axes?
"model_107/dense_428/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_107/dense_428/Tensordot/free?
#model_107/dense_428/Tensordot/ShapeShape)model_107/concatenate_161/concat:output:0*
T0*
_output_shapes
:2%
#model_107/dense_428/Tensordot/Shape?
+model_107/dense_428/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/dense_428/Tensordot/GatherV2/axis?
&model_107/dense_428/Tensordot/GatherV2GatherV2,model_107/dense_428/Tensordot/Shape:output:0+model_107/dense_428/Tensordot/free:output:04model_107/dense_428/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_107/dense_428/Tensordot/GatherV2?
-model_107/dense_428/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_107/dense_428/Tensordot/GatherV2_1/axis?
(model_107/dense_428/Tensordot/GatherV2_1GatherV2,model_107/dense_428/Tensordot/Shape:output:0+model_107/dense_428/Tensordot/axes:output:06model_107/dense_428/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_107/dense_428/Tensordot/GatherV2_1?
#model_107/dense_428/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_107/dense_428/Tensordot/Const?
"model_107/dense_428/Tensordot/ProdProd/model_107/dense_428/Tensordot/GatherV2:output:0,model_107/dense_428/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_107/dense_428/Tensordot/Prod?
%model_107/dense_428/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_107/dense_428/Tensordot/Const_1?
$model_107/dense_428/Tensordot/Prod_1Prod1model_107/dense_428/Tensordot/GatherV2_1:output:0.model_107/dense_428/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_107/dense_428/Tensordot/Prod_1?
)model_107/dense_428/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_107/dense_428/Tensordot/concat/axis?
$model_107/dense_428/Tensordot/concatConcatV2+model_107/dense_428/Tensordot/free:output:0+model_107/dense_428/Tensordot/axes:output:02model_107/dense_428/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_107/dense_428/Tensordot/concat?
#model_107/dense_428/Tensordot/stackPack+model_107/dense_428/Tensordot/Prod:output:0-model_107/dense_428/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_107/dense_428/Tensordot/stack?
'model_107/dense_428/Tensordot/transpose	Transpose)model_107/concatenate_161/concat:output:0-model_107/dense_428/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2)
'model_107/dense_428/Tensordot/transpose?
%model_107/dense_428/Tensordot/ReshapeReshape+model_107/dense_428/Tensordot/transpose:y:0,model_107/dense_428/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_107/dense_428/Tensordot/Reshape?
$model_107/dense_428/Tensordot/MatMulMatMul.model_107/dense_428/Tensordot/Reshape:output:04model_107/dense_428/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$model_107/dense_428/Tensordot/MatMul?
%model_107/dense_428/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_107/dense_428/Tensordot/Const_2?
+model_107/dense_428/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/dense_428/Tensordot/concat_1/axis?
&model_107/dense_428/Tensordot/concat_1ConcatV2/model_107/dense_428/Tensordot/GatherV2:output:0.model_107/dense_428/Tensordot/Const_2:output:04model_107/dense_428/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_107/dense_428/Tensordot/concat_1?
model_107/dense_428/TensordotReshape.model_107/dense_428/Tensordot/MatMul:product:0/model_107/dense_428/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
model_107/dense_428/Tensordot?
*model_107/dense_428/BiasAdd/ReadVariableOpReadVariableOp3model_107_dense_428_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*model_107/dense_428/BiasAdd/ReadVariableOp?
model_107/dense_428/BiasAddAdd&model_107/dense_428/Tensordot:output:02model_107/dense_428/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
model_107/dense_428/BiasAdd?
model_107/dense_428/ReluRelumodel_107/dense_428/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
model_107/dense_428/Relu?
)model_107/tf_op_layer_AddV2_107/AddV2_107AddV2&model_107/color_law/Tensordot:output:0Bmodel_107/tf_op_layer_strided_slice_433/strided_slice_433:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2+
)model_107/tf_op_layer_AddV2_107/AddV2_107?
,model_107/dense_429/Tensordot/ReadVariableOpReadVariableOp5model_107_dense_429_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02.
,model_107/dense_429/Tensordot/ReadVariableOp?
"model_107/dense_429/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_107/dense_429/Tensordot/axes?
"model_107/dense_429/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_107/dense_429/Tensordot/free?
#model_107/dense_429/Tensordot/ShapeShape&model_107/dense_428/Relu:activations:0*
T0*
_output_shapes
:2%
#model_107/dense_429/Tensordot/Shape?
+model_107/dense_429/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/dense_429/Tensordot/GatherV2/axis?
&model_107/dense_429/Tensordot/GatherV2GatherV2,model_107/dense_429/Tensordot/Shape:output:0+model_107/dense_429/Tensordot/free:output:04model_107/dense_429/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_107/dense_429/Tensordot/GatherV2?
-model_107/dense_429/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_107/dense_429/Tensordot/GatherV2_1/axis?
(model_107/dense_429/Tensordot/GatherV2_1GatherV2,model_107/dense_429/Tensordot/Shape:output:0+model_107/dense_429/Tensordot/axes:output:06model_107/dense_429/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_107/dense_429/Tensordot/GatherV2_1?
#model_107/dense_429/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_107/dense_429/Tensordot/Const?
"model_107/dense_429/Tensordot/ProdProd/model_107/dense_429/Tensordot/GatherV2:output:0,model_107/dense_429/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_107/dense_429/Tensordot/Prod?
%model_107/dense_429/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_107/dense_429/Tensordot/Const_1?
$model_107/dense_429/Tensordot/Prod_1Prod1model_107/dense_429/Tensordot/GatherV2_1:output:0.model_107/dense_429/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_107/dense_429/Tensordot/Prod_1?
)model_107/dense_429/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_107/dense_429/Tensordot/concat/axis?
$model_107/dense_429/Tensordot/concatConcatV2+model_107/dense_429/Tensordot/free:output:0+model_107/dense_429/Tensordot/axes:output:02model_107/dense_429/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_107/dense_429/Tensordot/concat?
#model_107/dense_429/Tensordot/stackPack+model_107/dense_429/Tensordot/Prod:output:0-model_107/dense_429/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_107/dense_429/Tensordot/stack?
'model_107/dense_429/Tensordot/transpose	Transpose&model_107/dense_428/Relu:activations:0-model_107/dense_429/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2)
'model_107/dense_429/Tensordot/transpose?
%model_107/dense_429/Tensordot/ReshapeReshape+model_107/dense_429/Tensordot/transpose:y:0,model_107/dense_429/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_107/dense_429/Tensordot/Reshape?
$model_107/dense_429/Tensordot/MatMulMatMul.model_107/dense_429/Tensordot/Reshape:output:04model_107/dense_429/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_107/dense_429/Tensordot/MatMul?
%model_107/dense_429/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_107/dense_429/Tensordot/Const_2?
+model_107/dense_429/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/dense_429/Tensordot/concat_1/axis?
&model_107/dense_429/Tensordot/concat_1ConcatV2/model_107/dense_429/Tensordot/GatherV2:output:0.model_107/dense_429/Tensordot/Const_2:output:04model_107/dense_429/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_107/dense_429/Tensordot/concat_1?
model_107/dense_429/TensordotReshape.model_107/dense_429/Tensordot/MatMul:product:0/model_107/dense_429/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_107/dense_429/Tensordot?
*model_107/dense_429/BiasAdd/ReadVariableOpReadVariableOp3model_107_dense_429_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_107/dense_429/BiasAdd/ReadVariableOp?
model_107/dense_429/BiasAddAdd&model_107/dense_429/Tensordot:output:02model_107/dense_429/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_107/dense_429/BiasAdd?
model_107/dense_429/ReluRelumodel_107/dense_429/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
model_107/dense_429/Relu?
'model_107/tf_op_layer_Mul_328/Mul_328/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2)
'model_107/tf_op_layer_Mul_328/Mul_328/x?
%model_107/tf_op_layer_Mul_328/Mul_328Mul0model_107/tf_op_layer_Mul_328/Mul_328/x:output:0-model_107/tf_op_layer_AddV2_107/AddV2_107:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_107/tf_op_layer_Mul_328/Mul_328?
,model_107/dense_430/Tensordot/ReadVariableOpReadVariableOp5model_107_dense_430_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,model_107/dense_430/Tensordot/ReadVariableOp?
"model_107/dense_430/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_107/dense_430/Tensordot/axes?
"model_107/dense_430/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_107/dense_430/Tensordot/free?
#model_107/dense_430/Tensordot/ShapeShape&model_107/dense_429/Relu:activations:0*
T0*
_output_shapes
:2%
#model_107/dense_430/Tensordot/Shape?
+model_107/dense_430/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/dense_430/Tensordot/GatherV2/axis?
&model_107/dense_430/Tensordot/GatherV2GatherV2,model_107/dense_430/Tensordot/Shape:output:0+model_107/dense_430/Tensordot/free:output:04model_107/dense_430/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_107/dense_430/Tensordot/GatherV2?
-model_107/dense_430/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_107/dense_430/Tensordot/GatherV2_1/axis?
(model_107/dense_430/Tensordot/GatherV2_1GatherV2,model_107/dense_430/Tensordot/Shape:output:0+model_107/dense_430/Tensordot/axes:output:06model_107/dense_430/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_107/dense_430/Tensordot/GatherV2_1?
#model_107/dense_430/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_107/dense_430/Tensordot/Const?
"model_107/dense_430/Tensordot/ProdProd/model_107/dense_430/Tensordot/GatherV2:output:0,model_107/dense_430/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_107/dense_430/Tensordot/Prod?
%model_107/dense_430/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_107/dense_430/Tensordot/Const_1?
$model_107/dense_430/Tensordot/Prod_1Prod1model_107/dense_430/Tensordot/GatherV2_1:output:0.model_107/dense_430/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_107/dense_430/Tensordot/Prod_1?
)model_107/dense_430/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_107/dense_430/Tensordot/concat/axis?
$model_107/dense_430/Tensordot/concatConcatV2+model_107/dense_430/Tensordot/free:output:0+model_107/dense_430/Tensordot/axes:output:02model_107/dense_430/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_107/dense_430/Tensordot/concat?
#model_107/dense_430/Tensordot/stackPack+model_107/dense_430/Tensordot/Prod:output:0-model_107/dense_430/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_107/dense_430/Tensordot/stack?
'model_107/dense_430/Tensordot/transpose	Transpose&model_107/dense_429/Relu:activations:0-model_107/dense_430/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2)
'model_107/dense_430/Tensordot/transpose?
%model_107/dense_430/Tensordot/ReshapeReshape+model_107/dense_430/Tensordot/transpose:y:0,model_107/dense_430/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_107/dense_430/Tensordot/Reshape?
$model_107/dense_430/Tensordot/MatMulMatMul.model_107/dense_430/Tensordot/Reshape:output:04model_107/dense_430/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_107/dense_430/Tensordot/MatMul?
%model_107/dense_430/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_107/dense_430/Tensordot/Const_2?
+model_107/dense_430/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/dense_430/Tensordot/concat_1/axis?
&model_107/dense_430/Tensordot/concat_1ConcatV2/model_107/dense_430/Tensordot/GatherV2:output:0.model_107/dense_430/Tensordot/Const_2:output:04model_107/dense_430/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_107/dense_430/Tensordot/concat_1?
model_107/dense_430/TensordotReshape.model_107/dense_430/Tensordot/MatMul:product:0/model_107/dense_430/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_107/dense_430/Tensordot?
*model_107/dense_430/BiasAdd/ReadVariableOpReadVariableOp3model_107_dense_430_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_107/dense_430/BiasAdd/ReadVariableOp?
model_107/dense_430/BiasAddAdd&model_107/dense_430/Tensordot:output:02model_107/dense_430/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_107/dense_430/BiasAdd?
model_107/dense_430/ReluRelumodel_107/dense_430/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
model_107/dense_430/Relu?
,model_107/dense_431/Tensordot/ReadVariableOpReadVariableOp5model_107_dense_431_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,model_107/dense_431/Tensordot/ReadVariableOp?
"model_107/dense_431/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_107/dense_431/Tensordot/axes?
"model_107/dense_431/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_107/dense_431/Tensordot/free?
#model_107/dense_431/Tensordot/ShapeShape&model_107/dense_430/Relu:activations:0*
T0*
_output_shapes
:2%
#model_107/dense_431/Tensordot/Shape?
+model_107/dense_431/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/dense_431/Tensordot/GatherV2/axis?
&model_107/dense_431/Tensordot/GatherV2GatherV2,model_107/dense_431/Tensordot/Shape:output:0+model_107/dense_431/Tensordot/free:output:04model_107/dense_431/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_107/dense_431/Tensordot/GatherV2?
-model_107/dense_431/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_107/dense_431/Tensordot/GatherV2_1/axis?
(model_107/dense_431/Tensordot/GatherV2_1GatherV2,model_107/dense_431/Tensordot/Shape:output:0+model_107/dense_431/Tensordot/axes:output:06model_107/dense_431/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_107/dense_431/Tensordot/GatherV2_1?
#model_107/dense_431/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_107/dense_431/Tensordot/Const?
"model_107/dense_431/Tensordot/ProdProd/model_107/dense_431/Tensordot/GatherV2:output:0,model_107/dense_431/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_107/dense_431/Tensordot/Prod?
%model_107/dense_431/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_107/dense_431/Tensordot/Const_1?
$model_107/dense_431/Tensordot/Prod_1Prod1model_107/dense_431/Tensordot/GatherV2_1:output:0.model_107/dense_431/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_107/dense_431/Tensordot/Prod_1?
)model_107/dense_431/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_107/dense_431/Tensordot/concat/axis?
$model_107/dense_431/Tensordot/concatConcatV2+model_107/dense_431/Tensordot/free:output:0+model_107/dense_431/Tensordot/axes:output:02model_107/dense_431/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_107/dense_431/Tensordot/concat?
#model_107/dense_431/Tensordot/stackPack+model_107/dense_431/Tensordot/Prod:output:0-model_107/dense_431/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_107/dense_431/Tensordot/stack?
'model_107/dense_431/Tensordot/transpose	Transpose&model_107/dense_430/Relu:activations:0-model_107/dense_431/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2)
'model_107/dense_431/Tensordot/transpose?
%model_107/dense_431/Tensordot/ReshapeReshape+model_107/dense_431/Tensordot/transpose:y:0,model_107/dense_431/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_107/dense_431/Tensordot/Reshape?
$model_107/dense_431/Tensordot/MatMulMatMul.model_107/dense_431/Tensordot/Reshape:output:04model_107/dense_431/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_107/dense_431/Tensordot/MatMul?
%model_107/dense_431/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_107/dense_431/Tensordot/Const_2?
+model_107/dense_431/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_107/dense_431/Tensordot/concat_1/axis?
&model_107/dense_431/Tensordot/concat_1ConcatV2/model_107/dense_431/Tensordot/GatherV2:output:0.model_107/dense_431/Tensordot/Const_2:output:04model_107/dense_431/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_107/dense_431/Tensordot/concat_1?
model_107/dense_431/TensordotReshape.model_107/dense_431/Tensordot/MatMul:product:0/model_107/dense_431/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_107/dense_431/Tensordot?
*model_107/dense_431/BiasAdd/ReadVariableOpReadVariableOp3model_107_dense_431_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_107/dense_431/BiasAdd/ReadVariableOp?
model_107/dense_431/BiasAddAdd&model_107/dense_431/Tensordot:output:02model_107/dense_431/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_107/dense_431/BiasAdd?
%model_107/tf_op_layer_Pow_53/Pow_53/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2'
%model_107/tf_op_layer_Pow_53/Pow_53/x?
#model_107/tf_op_layer_Pow_53/Pow_53Pow.model_107/tf_op_layer_Pow_53/Pow_53/x:output:0)model_107/tf_op_layer_Mul_328/Mul_328:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2%
#model_107/tf_op_layer_Pow_53/Pow_53?
%model_107/tf_op_layer_Mul_329/Mul_329Mulmodel_107/dense_431/BiasAdd:z:0'model_107/tf_op_layer_Pow_53/Pow_53:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_107/tf_op_layer_Mul_329/Mul_329?
%model_107/tf_op_layer_Relu_49/Relu_49Relu)model_107/tf_op_layer_Mul_329/Mul_329:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_107/tf_op_layer_Relu_49/Relu_49?
5model_107/tf_op_layer_Max_57/Max_57/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5model_107/tf_op_layer_Max_57/Max_57/reduction_indices?
#model_107/tf_op_layer_Max_57/Max_57Max	input_216>model_107/tf_op_layer_Max_57/Max_57/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2%
#model_107/tf_op_layer_Max_57/Max_57?
%model_107/tf_op_layer_Mul_330/Mul_330Mul3model_107/tf_op_layer_Relu_49/Relu_49:activations:0,model_107/tf_op_layer_Max_57/Max_57:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_107/tf_op_layer_Mul_330/Mul_330?
IdentityIdentity)model_107/tf_op_layer_Mul_330/Mul_330:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?::::::::::V R
'
_output_shapes
:?????????
'
_user_specified_namelatent_params:_[
+
_output_shapes
:????????? 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:????????? ?
#
_user_specified_name	input_216:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
E__inference_dense_429_layer_call_and_return_conditional_losses_446175

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2	
BiasAddX
ReluReluBiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????  :::S O
+
_output_shapes
:?????????  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
k
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_447214

inputs
identity[
	Mul_328/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
	Mul_328/x{
Mul_328MulMul_328/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_328d
IdentityIdentityMul_328:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
k
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_446197

inputs
identity[
	Mul_328/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
	Mul_328/x{
Mul_328MulMul_328/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_328d
IdentityIdentityMul_328:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
E__inference_color_law_layer_call_and_return_conditional_losses_446054

inputs%
!tensordot_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
	Tensordotk
IdentityIdentityTensordot:output:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: 
?
u
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_445956

inputs
identity?
strided_slice_432/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_432/begin
strided_slice_432/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_432/end?
strided_slice_432/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_432/strides?
strided_slice_432StridedSliceinputs strided_slice_432/begin:output:0strided_slice_432/end:output:0"strided_slice_432/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_432r
IdentityIdentitystrided_slice_432:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
y
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_446360

inputs
inputs_1
identityq
Mul_330Mulinputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_330d
IdentityIdentityMul_330:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:????????? ?:????????? :T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:SO
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_dense_428_layer_call_and_return_conditional_losses_447060

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2	
BiasAddW
ReluReluBiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? :::S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
}
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_447162
inputs_0
inputs_1
identityy
	AddV2_107AddV2inputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2
	AddV2_107f
IdentityIdentityAddV2_107:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:????????? ?:????????? :V R
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
{
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_445986

inputs
inputs_1
identityv
	AddV2_106AddV2inputsinputs_1*
T0*
_cloned(*+
_output_shapes
:????????? 2
	AddV2_106e
IdentityIdentityAddV2_106:z:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????? :????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:SO
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
w
K__inference_concatenate_161_layer_call_and_return_conditional_losses_447010
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????? :????????? :U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?

*__inference_dense_430_layer_call_fn_447208

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_430_layer_call_and_return_conditional_losses_4462362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :????????? ?::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
u
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_445972

inputs
identity?
strided_slice_435/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_435/begin
strided_slice_435/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_435/end?
strided_slice_435/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_435/strides?
strided_slice_435StridedSliceinputs strided_slice_435/begin:output:0strided_slice_435/end:output:0"strided_slice_435/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2
strided_slice_435r
IdentityIdentitystrided_slice_435:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
O
3__inference_tf_op_layer_Pow_53_layer_call_fn_447269

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_4463042
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
??
?
E__inference_model_107_layer_call_and_return_conditional_losses_446915
inputs_0
inputs_1
inputs_2/
+color_law_tensordot_readvariableop_resource/
+dense_428_tensordot_readvariableop_resource-
)dense_428_biasadd_readvariableop_resource/
+dense_429_tensordot_readvariableop_resource-
)dense_429_biasadd_readvariableop_resource/
+dense_430_tensordot_readvariableop_resource-
)dense_430_biasadd_readvariableop_resource/
+dense_431_tensordot_readvariableop_resource-
)dense_431_biasadd_readvariableop_resource
identity??
repeat_vector_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_53/ExpandDims/dim?
repeat_vector_53/ExpandDims
ExpandDimsinputs_0(repeat_vector_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
repeat_vector_53/ExpandDims?
repeat_vector_53/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
repeat_vector_53/stack?
repeat_vector_53/TileTile$repeat_vector_53/ExpandDims:output:0repeat_vector_53/stack:output:0*
T0*+
_output_shapes
:????????? 2
repeat_vector_53/Tile?
5tf_op_layer_strided_slice_432/strided_slice_432/beginConst*
_output_shapes
:*
dtype0*
valueB"        27
5tf_op_layer_strided_slice_432/strided_slice_432/begin?
3tf_op_layer_strided_slice_432/strided_slice_432/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_432/strided_slice_432/end?
7tf_op_layer_strided_slice_432/strided_slice_432/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_432/strided_slice_432/strides?
/tf_op_layer_strided_slice_432/strided_slice_432StridedSlicerepeat_vector_53/Tile:output:0>tf_op_layer_strided_slice_432/strided_slice_432/begin:output:0<tf_op_layer_strided_slice_432/strided_slice_432/end:output:0@tf_op_layer_strided_slice_432/strided_slice_432/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_432/strided_slice_432?
5tf_op_layer_strided_slice_435/strided_slice_435/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_435/strided_slice_435/begin?
3tf_op_layer_strided_slice_435/strided_slice_435/endConst*
_output_shapes
:*
dtype0*
valueB"        25
3tf_op_layer_strided_slice_435/strided_slice_435/end?
7tf_op_layer_strided_slice_435/strided_slice_435/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_435/strided_slice_435/strides?
/tf_op_layer_strided_slice_435/strided_slice_435StridedSlicerepeat_vector_53/Tile:output:0>tf_op_layer_strided_slice_435/strided_slice_435/begin:output:0<tf_op_layer_strided_slice_435/strided_slice_435/end:output:0@tf_op_layer_strided_slice_435/strided_slice_435/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask21
/tf_op_layer_strided_slice_435/strided_slice_435?
tf_op_layer_AddV2_106/AddV2_106AddV2inputs_18tf_op_layer_strided_slice_432/strided_slice_432:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2!
tf_op_layer_AddV2_106/AddV2_106?
5tf_op_layer_strided_slice_434/strided_slice_434/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_434/strided_slice_434/begin?
3tf_op_layer_strided_slice_434/strided_slice_434/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_434/strided_slice_434/end?
7tf_op_layer_strided_slice_434/strided_slice_434/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_434/strided_slice_434/strides?
/tf_op_layer_strided_slice_434/strided_slice_434StridedSlicerepeat_vector_53/Tile:output:0>tf_op_layer_strided_slice_434/strided_slice_434/begin:output:0<tf_op_layer_strided_slice_434/strided_slice_434/end:output:0@tf_op_layer_strided_slice_434/strided_slice_434/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_434/strided_slice_434|
concatenate_161/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_161/concat/axis?
concatenate_161/concatConcatV28tf_op_layer_strided_slice_435/strided_slice_435:output:0#tf_op_layer_AddV2_106/AddV2_106:z:0$concatenate_161/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concatenate_161/concat?
"color_law/Tensordot/ReadVariableOpReadVariableOp+color_law_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"color_law/Tensordot/ReadVariableOp~
color_law/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
color_law/Tensordot/axes?
color_law/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
color_law/Tensordot/free?
color_law/Tensordot/ShapeShape8tf_op_layer_strided_slice_434/strided_slice_434:output:0*
T0*
_output_shapes
:2
color_law/Tensordot/Shape?
!color_law/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!color_law/Tensordot/GatherV2/axis?
color_law/Tensordot/GatherV2GatherV2"color_law/Tensordot/Shape:output:0!color_law/Tensordot/free:output:0*color_law/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
color_law/Tensordot/GatherV2?
#color_law/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#color_law/Tensordot/GatherV2_1/axis?
color_law/Tensordot/GatherV2_1GatherV2"color_law/Tensordot/Shape:output:0!color_law/Tensordot/axes:output:0,color_law/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
color_law/Tensordot/GatherV2_1?
color_law/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
color_law/Tensordot/Const?
color_law/Tensordot/ProdProd%color_law/Tensordot/GatherV2:output:0"color_law/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
color_law/Tensordot/Prod?
color_law/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
color_law/Tensordot/Const_1?
color_law/Tensordot/Prod_1Prod'color_law/Tensordot/GatherV2_1:output:0$color_law/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
color_law/Tensordot/Prod_1?
color_law/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
color_law/Tensordot/concat/axis?
color_law/Tensordot/concatConcatV2!color_law/Tensordot/free:output:0!color_law/Tensordot/axes:output:0(color_law/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/concat?
color_law/Tensordot/stackPack!color_law/Tensordot/Prod:output:0#color_law/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/stack?
color_law/Tensordot/transpose	Transpose8tf_op_layer_strided_slice_434/strided_slice_434:output:0#color_law/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
color_law/Tensordot/transpose?
color_law/Tensordot/ReshapeReshape!color_law/Tensordot/transpose:y:0"color_law/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
color_law/Tensordot/Reshape?
color_law/Tensordot/MatMulMatMul$color_law/Tensordot/Reshape:output:0*color_law/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
color_law/Tensordot/MatMul?
color_law/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
color_law/Tensordot/Const_2?
!color_law/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!color_law/Tensordot/concat_1/axis?
color_law/Tensordot/concat_1ConcatV2%color_law/Tensordot/GatherV2:output:0$color_law/Tensordot/Const_2:output:0*color_law/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/concat_1?
color_law/TensordotReshape$color_law/Tensordot/MatMul:product:0%color_law/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
color_law/Tensordot?
5tf_op_layer_strided_slice_433/strided_slice_433/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_433/strided_slice_433/begin?
3tf_op_layer_strided_slice_433/strided_slice_433/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_433/strided_slice_433/end?
7tf_op_layer_strided_slice_433/strided_slice_433/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_433/strided_slice_433/strides?
/tf_op_layer_strided_slice_433/strided_slice_433StridedSlicerepeat_vector_53/Tile:output:0>tf_op_layer_strided_slice_433/strided_slice_433/begin:output:0<tf_op_layer_strided_slice_433/strided_slice_433/end:output:0@tf_op_layer_strided_slice_433/strided_slice_433/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_433/strided_slice_433?
"dense_428/Tensordot/ReadVariableOpReadVariableOp+dense_428_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02$
"dense_428/Tensordot/ReadVariableOp~
dense_428/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_428/Tensordot/axes?
dense_428/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_428/Tensordot/free?
dense_428/Tensordot/ShapeShapeconcatenate_161/concat:output:0*
T0*
_output_shapes
:2
dense_428/Tensordot/Shape?
!dense_428/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_428/Tensordot/GatherV2/axis?
dense_428/Tensordot/GatherV2GatherV2"dense_428/Tensordot/Shape:output:0!dense_428/Tensordot/free:output:0*dense_428/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_428/Tensordot/GatherV2?
#dense_428/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_428/Tensordot/GatherV2_1/axis?
dense_428/Tensordot/GatherV2_1GatherV2"dense_428/Tensordot/Shape:output:0!dense_428/Tensordot/axes:output:0,dense_428/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_428/Tensordot/GatherV2_1?
dense_428/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_428/Tensordot/Const?
dense_428/Tensordot/ProdProd%dense_428/Tensordot/GatherV2:output:0"dense_428/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_428/Tensordot/Prod?
dense_428/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_428/Tensordot/Const_1?
dense_428/Tensordot/Prod_1Prod'dense_428/Tensordot/GatherV2_1:output:0$dense_428/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_428/Tensordot/Prod_1?
dense_428/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_428/Tensordot/concat/axis?
dense_428/Tensordot/concatConcatV2!dense_428/Tensordot/free:output:0!dense_428/Tensordot/axes:output:0(dense_428/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_428/Tensordot/concat?
dense_428/Tensordot/stackPack!dense_428/Tensordot/Prod:output:0#dense_428/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_428/Tensordot/stack?
dense_428/Tensordot/transpose	Transposeconcatenate_161/concat:output:0#dense_428/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_428/Tensordot/transpose?
dense_428/Tensordot/ReshapeReshape!dense_428/Tensordot/transpose:y:0"dense_428/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_428/Tensordot/Reshape?
dense_428/Tensordot/MatMulMatMul$dense_428/Tensordot/Reshape:output:0*dense_428/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_428/Tensordot/MatMul?
dense_428/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_428/Tensordot/Const_2?
!dense_428/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_428/Tensordot/concat_1/axis?
dense_428/Tensordot/concat_1ConcatV2%dense_428/Tensordot/GatherV2:output:0$dense_428/Tensordot/Const_2:output:0*dense_428/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_428/Tensordot/concat_1?
dense_428/TensordotReshape$dense_428/Tensordot/MatMul:product:0%dense_428/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
dense_428/Tensordot?
 dense_428/BiasAdd/ReadVariableOpReadVariableOp)dense_428_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_428/BiasAdd/ReadVariableOp?
dense_428/BiasAddAdddense_428/Tensordot:output:0(dense_428/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
dense_428/BiasAddu
dense_428/ReluReludense_428/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
dense_428/Relu?
tf_op_layer_AddV2_107/AddV2_107AddV2color_law/Tensordot:output:08tf_op_layer_strided_slice_433/strided_slice_433:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2!
tf_op_layer_AddV2_107/AddV2_107?
"dense_429/Tensordot/ReadVariableOpReadVariableOp+dense_429_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"dense_429/Tensordot/ReadVariableOp~
dense_429/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_429/Tensordot/axes?
dense_429/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_429/Tensordot/free?
dense_429/Tensordot/ShapeShapedense_428/Relu:activations:0*
T0*
_output_shapes
:2
dense_429/Tensordot/Shape?
!dense_429/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_429/Tensordot/GatherV2/axis?
dense_429/Tensordot/GatherV2GatherV2"dense_429/Tensordot/Shape:output:0!dense_429/Tensordot/free:output:0*dense_429/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_429/Tensordot/GatherV2?
#dense_429/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_429/Tensordot/GatherV2_1/axis?
dense_429/Tensordot/GatherV2_1GatherV2"dense_429/Tensordot/Shape:output:0!dense_429/Tensordot/axes:output:0,dense_429/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_429/Tensordot/GatherV2_1?
dense_429/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_429/Tensordot/Const?
dense_429/Tensordot/ProdProd%dense_429/Tensordot/GatherV2:output:0"dense_429/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_429/Tensordot/Prod?
dense_429/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_429/Tensordot/Const_1?
dense_429/Tensordot/Prod_1Prod'dense_429/Tensordot/GatherV2_1:output:0$dense_429/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_429/Tensordot/Prod_1?
dense_429/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_429/Tensordot/concat/axis?
dense_429/Tensordot/concatConcatV2!dense_429/Tensordot/free:output:0!dense_429/Tensordot/axes:output:0(dense_429/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_429/Tensordot/concat?
dense_429/Tensordot/stackPack!dense_429/Tensordot/Prod:output:0#dense_429/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_429/Tensordot/stack?
dense_429/Tensordot/transpose	Transposedense_428/Relu:activations:0#dense_429/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2
dense_429/Tensordot/transpose?
dense_429/Tensordot/ReshapeReshape!dense_429/Tensordot/transpose:y:0"dense_429/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_429/Tensordot/Reshape?
dense_429/Tensordot/MatMulMatMul$dense_429/Tensordot/Reshape:output:0*dense_429/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_429/Tensordot/MatMul?
dense_429/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_429/Tensordot/Const_2?
!dense_429/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_429/Tensordot/concat_1/axis?
dense_429/Tensordot/concat_1ConcatV2%dense_429/Tensordot/GatherV2:output:0$dense_429/Tensordot/Const_2:output:0*dense_429/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_429/Tensordot/concat_1?
dense_429/TensordotReshape$dense_429/Tensordot/MatMul:product:0%dense_429/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_429/Tensordot?
 dense_429/BiasAdd/ReadVariableOpReadVariableOp)dense_429_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_429/BiasAdd/ReadVariableOp?
dense_429/BiasAddAdddense_429/Tensordot:output:0(dense_429/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_429/BiasAddv
dense_429/ReluReludense_429/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_429/Relu?
tf_op_layer_Mul_328/Mul_328/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
tf_op_layer_Mul_328/Mul_328/x?
tf_op_layer_Mul_328/Mul_328Mul&tf_op_layer_Mul_328/Mul_328/x:output:0#tf_op_layer_AddV2_107/AddV2_107:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_328/Mul_328?
"dense_430/Tensordot/ReadVariableOpReadVariableOp+dense_430_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_430/Tensordot/ReadVariableOp~
dense_430/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_430/Tensordot/axes?
dense_430/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_430/Tensordot/free?
dense_430/Tensordot/ShapeShapedense_429/Relu:activations:0*
T0*
_output_shapes
:2
dense_430/Tensordot/Shape?
!dense_430/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_430/Tensordot/GatherV2/axis?
dense_430/Tensordot/GatherV2GatherV2"dense_430/Tensordot/Shape:output:0!dense_430/Tensordot/free:output:0*dense_430/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_430/Tensordot/GatherV2?
#dense_430/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_430/Tensordot/GatherV2_1/axis?
dense_430/Tensordot/GatherV2_1GatherV2"dense_430/Tensordot/Shape:output:0!dense_430/Tensordot/axes:output:0,dense_430/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_430/Tensordot/GatherV2_1?
dense_430/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_430/Tensordot/Const?
dense_430/Tensordot/ProdProd%dense_430/Tensordot/GatherV2:output:0"dense_430/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_430/Tensordot/Prod?
dense_430/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_430/Tensordot/Const_1?
dense_430/Tensordot/Prod_1Prod'dense_430/Tensordot/GatherV2_1:output:0$dense_430/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_430/Tensordot/Prod_1?
dense_430/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_430/Tensordot/concat/axis?
dense_430/Tensordot/concatConcatV2!dense_430/Tensordot/free:output:0!dense_430/Tensordot/axes:output:0(dense_430/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_430/Tensordot/concat?
dense_430/Tensordot/stackPack!dense_430/Tensordot/Prod:output:0#dense_430/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_430/Tensordot/stack?
dense_430/Tensordot/transpose	Transposedense_429/Relu:activations:0#dense_430/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_430/Tensordot/transpose?
dense_430/Tensordot/ReshapeReshape!dense_430/Tensordot/transpose:y:0"dense_430/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_430/Tensordot/Reshape?
dense_430/Tensordot/MatMulMatMul$dense_430/Tensordot/Reshape:output:0*dense_430/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_430/Tensordot/MatMul?
dense_430/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_430/Tensordot/Const_2?
!dense_430/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_430/Tensordot/concat_1/axis?
dense_430/Tensordot/concat_1ConcatV2%dense_430/Tensordot/GatherV2:output:0$dense_430/Tensordot/Const_2:output:0*dense_430/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_430/Tensordot/concat_1?
dense_430/TensordotReshape$dense_430/Tensordot/MatMul:product:0%dense_430/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_430/Tensordot?
 dense_430/BiasAdd/ReadVariableOpReadVariableOp)dense_430_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_430/BiasAdd/ReadVariableOp?
dense_430/BiasAddAdddense_430/Tensordot:output:0(dense_430/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_430/BiasAddv
dense_430/ReluReludense_430/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_430/Relu?
"dense_431/Tensordot/ReadVariableOpReadVariableOp+dense_431_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_431/Tensordot/ReadVariableOp~
dense_431/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_431/Tensordot/axes?
dense_431/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_431/Tensordot/free?
dense_431/Tensordot/ShapeShapedense_430/Relu:activations:0*
T0*
_output_shapes
:2
dense_431/Tensordot/Shape?
!dense_431/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_431/Tensordot/GatherV2/axis?
dense_431/Tensordot/GatherV2GatherV2"dense_431/Tensordot/Shape:output:0!dense_431/Tensordot/free:output:0*dense_431/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_431/Tensordot/GatherV2?
#dense_431/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_431/Tensordot/GatherV2_1/axis?
dense_431/Tensordot/GatherV2_1GatherV2"dense_431/Tensordot/Shape:output:0!dense_431/Tensordot/axes:output:0,dense_431/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_431/Tensordot/GatherV2_1?
dense_431/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_431/Tensordot/Const?
dense_431/Tensordot/ProdProd%dense_431/Tensordot/GatherV2:output:0"dense_431/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_431/Tensordot/Prod?
dense_431/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_431/Tensordot/Const_1?
dense_431/Tensordot/Prod_1Prod'dense_431/Tensordot/GatherV2_1:output:0$dense_431/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_431/Tensordot/Prod_1?
dense_431/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_431/Tensordot/concat/axis?
dense_431/Tensordot/concatConcatV2!dense_431/Tensordot/free:output:0!dense_431/Tensordot/axes:output:0(dense_431/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_431/Tensordot/concat?
dense_431/Tensordot/stackPack!dense_431/Tensordot/Prod:output:0#dense_431/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_431/Tensordot/stack?
dense_431/Tensordot/transpose	Transposedense_430/Relu:activations:0#dense_431/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_431/Tensordot/transpose?
dense_431/Tensordot/ReshapeReshape!dense_431/Tensordot/transpose:y:0"dense_431/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_431/Tensordot/Reshape?
dense_431/Tensordot/MatMulMatMul$dense_431/Tensordot/Reshape:output:0*dense_431/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_431/Tensordot/MatMul?
dense_431/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_431/Tensordot/Const_2?
!dense_431/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_431/Tensordot/concat_1/axis?
dense_431/Tensordot/concat_1ConcatV2%dense_431/Tensordot/GatherV2:output:0$dense_431/Tensordot/Const_2:output:0*dense_431/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_431/Tensordot/concat_1?
dense_431/TensordotReshape$dense_431/Tensordot/MatMul:product:0%dense_431/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_431/Tensordot?
 dense_431/BiasAdd/ReadVariableOpReadVariableOp)dense_431_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_431/BiasAdd/ReadVariableOp?
dense_431/BiasAddAdddense_431/Tensordot:output:0(dense_431/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_431/BiasAdd
tf_op_layer_Pow_53/Pow_53/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
tf_op_layer_Pow_53/Pow_53/x?
tf_op_layer_Pow_53/Pow_53Pow$tf_op_layer_Pow_53/Pow_53/x:output:0tf_op_layer_Mul_328/Mul_328:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Pow_53/Pow_53?
tf_op_layer_Mul_329/Mul_329Muldense_431/BiasAdd:z:0tf_op_layer_Pow_53/Pow_53:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_329/Mul_329?
tf_op_layer_Relu_49/Relu_49Relutf_op_layer_Mul_329/Mul_329:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Relu_49/Relu_49?
+tf_op_layer_Max_57/Max_57/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+tf_op_layer_Max_57/Max_57/reduction_indices?
tf_op_layer_Max_57/Max_57Maxinputs_24tf_op_layer_Max_57/Max_57/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
tf_op_layer_Max_57/Max_57?
tf_op_layer_Mul_330/Mul_330Mul)tf_op_layer_Relu_49/Relu_49:activations:0"tf_op_layer_Max_57/Max_57:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_330/Mul_330x
IdentityIdentitytf_op_layer_Mul_330/Mul_330:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?::::::::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
{
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_447275
inputs_0
inputs_1
identitys
Mul_329Mulinputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_329d
IdentityIdentityMul_329:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:????????? ?:????????? ?:V R
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/1
?
?
*__inference_model_107_layer_call_fn_446550
latent_params
conditional_params
	input_216
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_216unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_model_107_layer_call_and_return_conditional_losses_4465292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namelatent_params:_[
+
_output_shapes
:????????? 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:????????? ?
#
_user_specified_name	input_216:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
u
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_446986

inputs
identity?
strided_slice_435/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_435/begin
strided_slice_435/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_435/end?
strided_slice_435/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_435/strides?
strided_slice_435StridedSliceinputs strided_slice_435/begin:output:0strided_slice_435/end:output:0"strided_slice_435/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2
strided_slice_435r
IdentityIdentitystrided_slice_435:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
Z
>__inference_tf_op_layer_strided_slice_435_layer_call_fn_446991

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_4459722
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

*__inference_dense_431_layer_call_fn_447258

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_431_layer_call_and_return_conditional_losses_4462822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :????????? ?::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
p
*__inference_color_law_layer_call_fn_447103

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4460542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: 
?

*__inference_dense_429_layer_call_fn_447156

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_429_layer_call_and_return_conditional_losses_4461752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_dense_431_layer_call_and_return_conditional_losses_447249

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2	
BiasAddd
IdentityIdentityBiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :????????? ?:::T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
y
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_446318

inputs
inputs_1
identityq
Mul_329Mulinputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_329d
IdentityIdentityMul_329:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:????????? ?:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:TP
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
j
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_447264

inputs
identityY
Pow_53/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

Pow_53/xx
Pow_53PowPow_53/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2
Pow_53c
IdentityIdentity
Pow_53:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
\
0__inference_concatenate_161_layer_call_fn_447016
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_161_layer_call_and_return_conditional_losses_4460182
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????? :????????? :U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
?
E__inference_dense_431_layer_call_and_return_conditional_losses_446282

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2	
BiasAddd
IdentityIdentityBiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :????????? ?:::T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
u
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_446973

inputs
identity?
strided_slice_432/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_432/begin
strided_slice_432/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_432/end?
strided_slice_432/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_432/strides?
strided_slice_432StridedSliceinputs strided_slice_432/begin:output:0strided_slice_432/end:output:0"strided_slice_432/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_432r
IdentityIdentitystrided_slice_432:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
h
L__inference_repeat_vector_53_layer_call_and_return_conditional_losses_445935

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
stackx
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :????????? ?????????2
Tilen
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :????????? ?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_446577
conditional_params
	input_216
latent_params
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_216unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_4459262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:????????? :????????? ?:?????????:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:????????? 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:????????? ?
#
_user_specified_name	input_216:VR
'
_output_shapes
:?????????
'
_user_specified_namelatent_params:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
??
?
E__inference_model_107_layer_call_and_return_conditional_losses_446746
inputs_0
inputs_1
inputs_2/
+color_law_tensordot_readvariableop_resource/
+dense_428_tensordot_readvariableop_resource-
)dense_428_biasadd_readvariableop_resource/
+dense_429_tensordot_readvariableop_resource-
)dense_429_biasadd_readvariableop_resource/
+dense_430_tensordot_readvariableop_resource-
)dense_430_biasadd_readvariableop_resource/
+dense_431_tensordot_readvariableop_resource-
)dense_431_biasadd_readvariableop_resource
identity??
repeat_vector_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_53/ExpandDims/dim?
repeat_vector_53/ExpandDims
ExpandDimsinputs_0(repeat_vector_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
repeat_vector_53/ExpandDims?
repeat_vector_53/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
repeat_vector_53/stack?
repeat_vector_53/TileTile$repeat_vector_53/ExpandDims:output:0repeat_vector_53/stack:output:0*
T0*+
_output_shapes
:????????? 2
repeat_vector_53/Tile?
5tf_op_layer_strided_slice_432/strided_slice_432/beginConst*
_output_shapes
:*
dtype0*
valueB"        27
5tf_op_layer_strided_slice_432/strided_slice_432/begin?
3tf_op_layer_strided_slice_432/strided_slice_432/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_432/strided_slice_432/end?
7tf_op_layer_strided_slice_432/strided_slice_432/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_432/strided_slice_432/strides?
/tf_op_layer_strided_slice_432/strided_slice_432StridedSlicerepeat_vector_53/Tile:output:0>tf_op_layer_strided_slice_432/strided_slice_432/begin:output:0<tf_op_layer_strided_slice_432/strided_slice_432/end:output:0@tf_op_layer_strided_slice_432/strided_slice_432/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_432/strided_slice_432?
5tf_op_layer_strided_slice_435/strided_slice_435/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_435/strided_slice_435/begin?
3tf_op_layer_strided_slice_435/strided_slice_435/endConst*
_output_shapes
:*
dtype0*
valueB"        25
3tf_op_layer_strided_slice_435/strided_slice_435/end?
7tf_op_layer_strided_slice_435/strided_slice_435/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_435/strided_slice_435/strides?
/tf_op_layer_strided_slice_435/strided_slice_435StridedSlicerepeat_vector_53/Tile:output:0>tf_op_layer_strided_slice_435/strided_slice_435/begin:output:0<tf_op_layer_strided_slice_435/strided_slice_435/end:output:0@tf_op_layer_strided_slice_435/strided_slice_435/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask21
/tf_op_layer_strided_slice_435/strided_slice_435?
tf_op_layer_AddV2_106/AddV2_106AddV2inputs_18tf_op_layer_strided_slice_432/strided_slice_432:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2!
tf_op_layer_AddV2_106/AddV2_106?
5tf_op_layer_strided_slice_434/strided_slice_434/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_434/strided_slice_434/begin?
3tf_op_layer_strided_slice_434/strided_slice_434/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_434/strided_slice_434/end?
7tf_op_layer_strided_slice_434/strided_slice_434/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_434/strided_slice_434/strides?
/tf_op_layer_strided_slice_434/strided_slice_434StridedSlicerepeat_vector_53/Tile:output:0>tf_op_layer_strided_slice_434/strided_slice_434/begin:output:0<tf_op_layer_strided_slice_434/strided_slice_434/end:output:0@tf_op_layer_strided_slice_434/strided_slice_434/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_434/strided_slice_434|
concatenate_161/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_161/concat/axis?
concatenate_161/concatConcatV28tf_op_layer_strided_slice_435/strided_slice_435:output:0#tf_op_layer_AddV2_106/AddV2_106:z:0$concatenate_161/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concatenate_161/concat?
"color_law/Tensordot/ReadVariableOpReadVariableOp+color_law_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"color_law/Tensordot/ReadVariableOp~
color_law/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
color_law/Tensordot/axes?
color_law/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
color_law/Tensordot/free?
color_law/Tensordot/ShapeShape8tf_op_layer_strided_slice_434/strided_slice_434:output:0*
T0*
_output_shapes
:2
color_law/Tensordot/Shape?
!color_law/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!color_law/Tensordot/GatherV2/axis?
color_law/Tensordot/GatherV2GatherV2"color_law/Tensordot/Shape:output:0!color_law/Tensordot/free:output:0*color_law/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
color_law/Tensordot/GatherV2?
#color_law/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#color_law/Tensordot/GatherV2_1/axis?
color_law/Tensordot/GatherV2_1GatherV2"color_law/Tensordot/Shape:output:0!color_law/Tensordot/axes:output:0,color_law/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
color_law/Tensordot/GatherV2_1?
color_law/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
color_law/Tensordot/Const?
color_law/Tensordot/ProdProd%color_law/Tensordot/GatherV2:output:0"color_law/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
color_law/Tensordot/Prod?
color_law/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
color_law/Tensordot/Const_1?
color_law/Tensordot/Prod_1Prod'color_law/Tensordot/GatherV2_1:output:0$color_law/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
color_law/Tensordot/Prod_1?
color_law/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
color_law/Tensordot/concat/axis?
color_law/Tensordot/concatConcatV2!color_law/Tensordot/free:output:0!color_law/Tensordot/axes:output:0(color_law/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/concat?
color_law/Tensordot/stackPack!color_law/Tensordot/Prod:output:0#color_law/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/stack?
color_law/Tensordot/transpose	Transpose8tf_op_layer_strided_slice_434/strided_slice_434:output:0#color_law/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
color_law/Tensordot/transpose?
color_law/Tensordot/ReshapeReshape!color_law/Tensordot/transpose:y:0"color_law/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
color_law/Tensordot/Reshape?
color_law/Tensordot/MatMulMatMul$color_law/Tensordot/Reshape:output:0*color_law/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
color_law/Tensordot/MatMul?
color_law/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
color_law/Tensordot/Const_2?
!color_law/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!color_law/Tensordot/concat_1/axis?
color_law/Tensordot/concat_1ConcatV2%color_law/Tensordot/GatherV2:output:0$color_law/Tensordot/Const_2:output:0*color_law/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/concat_1?
color_law/TensordotReshape$color_law/Tensordot/MatMul:product:0%color_law/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
color_law/Tensordot?
5tf_op_layer_strided_slice_433/strided_slice_433/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_433/strided_slice_433/begin?
3tf_op_layer_strided_slice_433/strided_slice_433/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_433/strided_slice_433/end?
7tf_op_layer_strided_slice_433/strided_slice_433/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_433/strided_slice_433/strides?
/tf_op_layer_strided_slice_433/strided_slice_433StridedSlicerepeat_vector_53/Tile:output:0>tf_op_layer_strided_slice_433/strided_slice_433/begin:output:0<tf_op_layer_strided_slice_433/strided_slice_433/end:output:0@tf_op_layer_strided_slice_433/strided_slice_433/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_433/strided_slice_433?
"dense_428/Tensordot/ReadVariableOpReadVariableOp+dense_428_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02$
"dense_428/Tensordot/ReadVariableOp~
dense_428/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_428/Tensordot/axes?
dense_428/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_428/Tensordot/free?
dense_428/Tensordot/ShapeShapeconcatenate_161/concat:output:0*
T0*
_output_shapes
:2
dense_428/Tensordot/Shape?
!dense_428/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_428/Tensordot/GatherV2/axis?
dense_428/Tensordot/GatherV2GatherV2"dense_428/Tensordot/Shape:output:0!dense_428/Tensordot/free:output:0*dense_428/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_428/Tensordot/GatherV2?
#dense_428/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_428/Tensordot/GatherV2_1/axis?
dense_428/Tensordot/GatherV2_1GatherV2"dense_428/Tensordot/Shape:output:0!dense_428/Tensordot/axes:output:0,dense_428/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_428/Tensordot/GatherV2_1?
dense_428/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_428/Tensordot/Const?
dense_428/Tensordot/ProdProd%dense_428/Tensordot/GatherV2:output:0"dense_428/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_428/Tensordot/Prod?
dense_428/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_428/Tensordot/Const_1?
dense_428/Tensordot/Prod_1Prod'dense_428/Tensordot/GatherV2_1:output:0$dense_428/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_428/Tensordot/Prod_1?
dense_428/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_428/Tensordot/concat/axis?
dense_428/Tensordot/concatConcatV2!dense_428/Tensordot/free:output:0!dense_428/Tensordot/axes:output:0(dense_428/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_428/Tensordot/concat?
dense_428/Tensordot/stackPack!dense_428/Tensordot/Prod:output:0#dense_428/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_428/Tensordot/stack?
dense_428/Tensordot/transpose	Transposeconcatenate_161/concat:output:0#dense_428/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_428/Tensordot/transpose?
dense_428/Tensordot/ReshapeReshape!dense_428/Tensordot/transpose:y:0"dense_428/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_428/Tensordot/Reshape?
dense_428/Tensordot/MatMulMatMul$dense_428/Tensordot/Reshape:output:0*dense_428/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_428/Tensordot/MatMul?
dense_428/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_428/Tensordot/Const_2?
!dense_428/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_428/Tensordot/concat_1/axis?
dense_428/Tensordot/concat_1ConcatV2%dense_428/Tensordot/GatherV2:output:0$dense_428/Tensordot/Const_2:output:0*dense_428/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_428/Tensordot/concat_1?
dense_428/TensordotReshape$dense_428/Tensordot/MatMul:product:0%dense_428/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
dense_428/Tensordot?
 dense_428/BiasAdd/ReadVariableOpReadVariableOp)dense_428_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_428/BiasAdd/ReadVariableOp?
dense_428/BiasAddAdddense_428/Tensordot:output:0(dense_428/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
dense_428/BiasAddu
dense_428/ReluReludense_428/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
dense_428/Relu?
tf_op_layer_AddV2_107/AddV2_107AddV2color_law/Tensordot:output:08tf_op_layer_strided_slice_433/strided_slice_433:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2!
tf_op_layer_AddV2_107/AddV2_107?
"dense_429/Tensordot/ReadVariableOpReadVariableOp+dense_429_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"dense_429/Tensordot/ReadVariableOp~
dense_429/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_429/Tensordot/axes?
dense_429/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_429/Tensordot/free?
dense_429/Tensordot/ShapeShapedense_428/Relu:activations:0*
T0*
_output_shapes
:2
dense_429/Tensordot/Shape?
!dense_429/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_429/Tensordot/GatherV2/axis?
dense_429/Tensordot/GatherV2GatherV2"dense_429/Tensordot/Shape:output:0!dense_429/Tensordot/free:output:0*dense_429/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_429/Tensordot/GatherV2?
#dense_429/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_429/Tensordot/GatherV2_1/axis?
dense_429/Tensordot/GatherV2_1GatherV2"dense_429/Tensordot/Shape:output:0!dense_429/Tensordot/axes:output:0,dense_429/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_429/Tensordot/GatherV2_1?
dense_429/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_429/Tensordot/Const?
dense_429/Tensordot/ProdProd%dense_429/Tensordot/GatherV2:output:0"dense_429/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_429/Tensordot/Prod?
dense_429/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_429/Tensordot/Const_1?
dense_429/Tensordot/Prod_1Prod'dense_429/Tensordot/GatherV2_1:output:0$dense_429/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_429/Tensordot/Prod_1?
dense_429/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_429/Tensordot/concat/axis?
dense_429/Tensordot/concatConcatV2!dense_429/Tensordot/free:output:0!dense_429/Tensordot/axes:output:0(dense_429/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_429/Tensordot/concat?
dense_429/Tensordot/stackPack!dense_429/Tensordot/Prod:output:0#dense_429/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_429/Tensordot/stack?
dense_429/Tensordot/transpose	Transposedense_428/Relu:activations:0#dense_429/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2
dense_429/Tensordot/transpose?
dense_429/Tensordot/ReshapeReshape!dense_429/Tensordot/transpose:y:0"dense_429/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_429/Tensordot/Reshape?
dense_429/Tensordot/MatMulMatMul$dense_429/Tensordot/Reshape:output:0*dense_429/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_429/Tensordot/MatMul?
dense_429/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_429/Tensordot/Const_2?
!dense_429/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_429/Tensordot/concat_1/axis?
dense_429/Tensordot/concat_1ConcatV2%dense_429/Tensordot/GatherV2:output:0$dense_429/Tensordot/Const_2:output:0*dense_429/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_429/Tensordot/concat_1?
dense_429/TensordotReshape$dense_429/Tensordot/MatMul:product:0%dense_429/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_429/Tensordot?
 dense_429/BiasAdd/ReadVariableOpReadVariableOp)dense_429_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_429/BiasAdd/ReadVariableOp?
dense_429/BiasAddAdddense_429/Tensordot:output:0(dense_429/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_429/BiasAddv
dense_429/ReluReludense_429/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_429/Relu?
tf_op_layer_Mul_328/Mul_328/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
tf_op_layer_Mul_328/Mul_328/x?
tf_op_layer_Mul_328/Mul_328Mul&tf_op_layer_Mul_328/Mul_328/x:output:0#tf_op_layer_AddV2_107/AddV2_107:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_328/Mul_328?
"dense_430/Tensordot/ReadVariableOpReadVariableOp+dense_430_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_430/Tensordot/ReadVariableOp~
dense_430/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_430/Tensordot/axes?
dense_430/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_430/Tensordot/free?
dense_430/Tensordot/ShapeShapedense_429/Relu:activations:0*
T0*
_output_shapes
:2
dense_430/Tensordot/Shape?
!dense_430/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_430/Tensordot/GatherV2/axis?
dense_430/Tensordot/GatherV2GatherV2"dense_430/Tensordot/Shape:output:0!dense_430/Tensordot/free:output:0*dense_430/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_430/Tensordot/GatherV2?
#dense_430/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_430/Tensordot/GatherV2_1/axis?
dense_430/Tensordot/GatherV2_1GatherV2"dense_430/Tensordot/Shape:output:0!dense_430/Tensordot/axes:output:0,dense_430/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_430/Tensordot/GatherV2_1?
dense_430/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_430/Tensordot/Const?
dense_430/Tensordot/ProdProd%dense_430/Tensordot/GatherV2:output:0"dense_430/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_430/Tensordot/Prod?
dense_430/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_430/Tensordot/Const_1?
dense_430/Tensordot/Prod_1Prod'dense_430/Tensordot/GatherV2_1:output:0$dense_430/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_430/Tensordot/Prod_1?
dense_430/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_430/Tensordot/concat/axis?
dense_430/Tensordot/concatConcatV2!dense_430/Tensordot/free:output:0!dense_430/Tensordot/axes:output:0(dense_430/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_430/Tensordot/concat?
dense_430/Tensordot/stackPack!dense_430/Tensordot/Prod:output:0#dense_430/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_430/Tensordot/stack?
dense_430/Tensordot/transpose	Transposedense_429/Relu:activations:0#dense_430/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_430/Tensordot/transpose?
dense_430/Tensordot/ReshapeReshape!dense_430/Tensordot/transpose:y:0"dense_430/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_430/Tensordot/Reshape?
dense_430/Tensordot/MatMulMatMul$dense_430/Tensordot/Reshape:output:0*dense_430/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_430/Tensordot/MatMul?
dense_430/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_430/Tensordot/Const_2?
!dense_430/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_430/Tensordot/concat_1/axis?
dense_430/Tensordot/concat_1ConcatV2%dense_430/Tensordot/GatherV2:output:0$dense_430/Tensordot/Const_2:output:0*dense_430/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_430/Tensordot/concat_1?
dense_430/TensordotReshape$dense_430/Tensordot/MatMul:product:0%dense_430/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_430/Tensordot?
 dense_430/BiasAdd/ReadVariableOpReadVariableOp)dense_430_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_430/BiasAdd/ReadVariableOp?
dense_430/BiasAddAdddense_430/Tensordot:output:0(dense_430/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_430/BiasAddv
dense_430/ReluReludense_430/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_430/Relu?
"dense_431/Tensordot/ReadVariableOpReadVariableOp+dense_431_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_431/Tensordot/ReadVariableOp~
dense_431/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_431/Tensordot/axes?
dense_431/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_431/Tensordot/free?
dense_431/Tensordot/ShapeShapedense_430/Relu:activations:0*
T0*
_output_shapes
:2
dense_431/Tensordot/Shape?
!dense_431/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_431/Tensordot/GatherV2/axis?
dense_431/Tensordot/GatherV2GatherV2"dense_431/Tensordot/Shape:output:0!dense_431/Tensordot/free:output:0*dense_431/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_431/Tensordot/GatherV2?
#dense_431/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_431/Tensordot/GatherV2_1/axis?
dense_431/Tensordot/GatherV2_1GatherV2"dense_431/Tensordot/Shape:output:0!dense_431/Tensordot/axes:output:0,dense_431/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_431/Tensordot/GatherV2_1?
dense_431/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_431/Tensordot/Const?
dense_431/Tensordot/ProdProd%dense_431/Tensordot/GatherV2:output:0"dense_431/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_431/Tensordot/Prod?
dense_431/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_431/Tensordot/Const_1?
dense_431/Tensordot/Prod_1Prod'dense_431/Tensordot/GatherV2_1:output:0$dense_431/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_431/Tensordot/Prod_1?
dense_431/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_431/Tensordot/concat/axis?
dense_431/Tensordot/concatConcatV2!dense_431/Tensordot/free:output:0!dense_431/Tensordot/axes:output:0(dense_431/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_431/Tensordot/concat?
dense_431/Tensordot/stackPack!dense_431/Tensordot/Prod:output:0#dense_431/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_431/Tensordot/stack?
dense_431/Tensordot/transpose	Transposedense_430/Relu:activations:0#dense_431/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_431/Tensordot/transpose?
dense_431/Tensordot/ReshapeReshape!dense_431/Tensordot/transpose:y:0"dense_431/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_431/Tensordot/Reshape?
dense_431/Tensordot/MatMulMatMul$dense_431/Tensordot/Reshape:output:0*dense_431/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_431/Tensordot/MatMul?
dense_431/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_431/Tensordot/Const_2?
!dense_431/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_431/Tensordot/concat_1/axis?
dense_431/Tensordot/concat_1ConcatV2%dense_431/Tensordot/GatherV2:output:0$dense_431/Tensordot/Const_2:output:0*dense_431/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_431/Tensordot/concat_1?
dense_431/TensordotReshape$dense_431/Tensordot/MatMul:product:0%dense_431/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_431/Tensordot?
 dense_431/BiasAdd/ReadVariableOpReadVariableOp)dense_431_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_431/BiasAdd/ReadVariableOp?
dense_431/BiasAddAdddense_431/Tensordot:output:0(dense_431/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_431/BiasAdd
tf_op_layer_Pow_53/Pow_53/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
tf_op_layer_Pow_53/Pow_53/x?
tf_op_layer_Pow_53/Pow_53Pow$tf_op_layer_Pow_53/Pow_53/x:output:0tf_op_layer_Mul_328/Mul_328:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Pow_53/Pow_53?
tf_op_layer_Mul_329/Mul_329Muldense_431/BiasAdd:z:0tf_op_layer_Pow_53/Pow_53:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_329/Mul_329?
tf_op_layer_Relu_49/Relu_49Relutf_op_layer_Mul_329/Mul_329:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Relu_49/Relu_49?
+tf_op_layer_Max_57/Max_57/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+tf_op_layer_Max_57/Max_57/reduction_indices?
tf_op_layer_Max_57/Max_57Maxinputs_24tf_op_layer_Max_57/Max_57/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
tf_op_layer_Max_57/Max_57?
tf_op_layer_Mul_330/Mul_330Mul)tf_op_layer_Relu_49/Relu_49:activations:0"tf_op_layer_Max_57/Max_57:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_330/Mul_330x
IdentityIdentitytf_op_layer_Mul_330/Mul_330:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?::::::::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?K
?
E__inference_model_107_layer_call_and_return_conditional_losses_446370
latent_params
conditional_params
	input_216
color_law_446063
dense_428_446124
dense_428_446126
dense_429_446186
dense_429_446188
dense_430_446247
dense_430_446249
dense_431_446293
dense_431_446295
identity??!color_law/StatefulPartitionedCall?!dense_428/StatefulPartitionedCall?!dense_429/StatefulPartitionedCall?!dense_430/StatefulPartitionedCall?!dense_431/StatefulPartitionedCall?
 repeat_vector_53/PartitionedCallPartitionedCalllatent_params*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_repeat_vector_53_layer_call_and_return_conditional_losses_4459352"
 repeat_vector_53/PartitionedCall?
-tf_op_layer_strided_slice_432/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_4459562/
-tf_op_layer_strided_slice_432/PartitionedCall?
-tf_op_layer_strided_slice_435/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_4459722/
-tf_op_layer_strided_slice_435/PartitionedCall?
%tf_op_layer_AddV2_106/PartitionedCallPartitionedCallconditional_params6tf_op_layer_strided_slice_432/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_4459862'
%tf_op_layer_AddV2_106/PartitionedCall?
-tf_op_layer_strided_slice_434/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_4460032/
-tf_op_layer_strided_slice_434/PartitionedCall?
concatenate_161/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_435/PartitionedCall:output:0.tf_op_layer_AddV2_106/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_161_layer_call_and_return_conditional_losses_4460182!
concatenate_161/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_434/PartitionedCall:output:0color_law_446063*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4460542#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_433/PartitionedCallPartitionedCall)repeat_vector_53/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_4460742/
-tf_op_layer_strided_slice_433/PartitionedCall?
!dense_428/StatefulPartitionedCallStatefulPartitionedCall(concatenate_161/PartitionedCall:output:0dense_428_446124dense_428_446126*
Tin
2*
Tout
2*+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_428_layer_call_and_return_conditional_losses_4461132#
!dense_428/StatefulPartitionedCall?
%tf_op_layer_AddV2_107/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_433/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_4461352'
%tf_op_layer_AddV2_107/PartitionedCall?
!dense_429/StatefulPartitionedCallStatefulPartitionedCall*dense_428/StatefulPartitionedCall:output:0dense_429_446186dense_429_446188*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_429_layer_call_and_return_conditional_losses_4461752#
!dense_429/StatefulPartitionedCall?
#tf_op_layer_Mul_328/PartitionedCallPartitionedCall.tf_op_layer_AddV2_107/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_4461972%
#tf_op_layer_Mul_328/PartitionedCall?
!dense_430/StatefulPartitionedCallStatefulPartitionedCall*dense_429/StatefulPartitionedCall:output:0dense_430_446247dense_430_446249*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_430_layer_call_and_return_conditional_losses_4462362#
!dense_430/StatefulPartitionedCall?
!dense_431/StatefulPartitionedCallStatefulPartitionedCall*dense_430/StatefulPartitionedCall:output:0dense_431_446293dense_431_446295*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_431_layer_call_and_return_conditional_losses_4462822#
!dense_431/StatefulPartitionedCall?
"tf_op_layer_Pow_53/PartitionedCallPartitionedCall,tf_op_layer_Mul_328/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_4463042$
"tf_op_layer_Pow_53/PartitionedCall?
#tf_op_layer_Mul_329/PartitionedCallPartitionedCall*dense_431/StatefulPartitionedCall:output:0+tf_op_layer_Pow_53/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_4463182%
#tf_op_layer_Mul_329/PartitionedCall?
#tf_op_layer_Relu_49/PartitionedCallPartitionedCall,tf_op_layer_Mul_329/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_4463322%
#tf_op_layer_Relu_49/PartitionedCall?
"tf_op_layer_Max_57/PartitionedCallPartitionedCall	input_216*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_4463462$
"tf_op_layer_Max_57/PartitionedCall?
#tf_op_layer_Mul_330/PartitionedCallPartitionedCall,tf_op_layer_Relu_49/PartitionedCall:output:0+tf_op_layer_Max_57/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4463602%
#tf_op_layer_Mul_330/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_330/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dense_429/StatefulPartitionedCall"^dense_430/StatefulPartitionedCall"^dense_431/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dense_429/StatefulPartitionedCall!dense_429/StatefulPartitionedCall2F
!dense_430/StatefulPartitionedCall!dense_430/StatefulPartitionedCall2F
!dense_431/StatefulPartitionedCall!dense_431/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namelatent_params:_[
+
_output_shapes
:????????? 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:????????? ?
#
_user_specified_name	input_216:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
`
4__inference_tf_op_layer_Mul_329_layer_call_fn_447281
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_4463182
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:????????? ?:????????? ?:V R
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/1
?
{
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_447308
inputs_0
inputs_1
identitys
Mul_330Mulinputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_330d
IdentityIdentityMul_330:z:0*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:????????? ?:????????? :V R
,
_output_shapes
:????????? ?
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
b
6__inference_tf_op_layer_AddV2_106_layer_call_fn_447003
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_4459862
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????? :????????? :U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
conditional_params?
$serving_default_conditional_params:0????????? 
D
	input_2167
serving_default_input_216:0????????? ?
G
latent_params6
serving_default_latent_params:0?????????L
tf_op_layer_Mul_3305
StatefulPartitionedCall:0????????? ?tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-0
	layer-8

layer_with_weights-1

layer-9
layer-10
layer_with_weights-2
layer-11
layer-12
layer_with_weights-3
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_model??{"class_name": "Model", "name": "model_107", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_107", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}, "name": "latent_params", "inbound_nodes": []}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector_53", "trainable": true, "dtype": "float32", "n": 32}, "name": "repeat_vector_53", "inbound_nodes": [[["latent_params", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}, "name": "conditional_params", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_432", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_432", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_432/begin", "strided_slice_432/end", "strided_slice_432/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_432", "inbound_nodes": [[["repeat_vector_53", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_435", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_435", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_435/begin", "strided_slice_435/end", "strided_slice_435/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "2"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_435", "inbound_nodes": [[["repeat_vector_53", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_106", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_106", "op": "AddV2", "input": ["conditional_params_56", "strided_slice_432"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_106", "inbound_nodes": [[["conditional_params", 0, 0, {}], ["tf_op_layer_strided_slice_432", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_161", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_161", "inbound_nodes": [[["tf_op_layer_strided_slice_435", 0, 0, {}], ["tf_op_layer_AddV2_106", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_434", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_434", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_434/begin", "strided_slice_434/end", "strided_slice_434/strides"], "attr": {"shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_434", "inbound_nodes": [[["repeat_vector_53", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_428", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_428", "inbound_nodes": [[["concatenate_161", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "color_law", "inbound_nodes": [[["tf_op_layer_strided_slice_434", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_433", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_433", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_433/begin", "strided_slice_433/end", "strided_slice_433/strides"], "attr": {"Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_433", "inbound_nodes": [[["repeat_vector_53", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_429", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_429", "inbound_nodes": [[["dense_428", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_107", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_107", "op": "AddV2", "input": ["color_law_56/Identity", "strided_slice_433"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_107", "inbound_nodes": [[["color_law", 0, 0, {}], ["tf_op_layer_strided_slice_433", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_430", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_430", "inbound_nodes": [[["dense_429", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_328", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_328", "op": "Mul", "input": ["Mul_328/x", "AddV2_107"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}, "name": "tf_op_layer_Mul_328", "inbound_nodes": [[["tf_op_layer_AddV2_107", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_431", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_431", "inbound_nodes": [[["dense_430", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow_53", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_53", "op": "Pow", "input": ["Pow_53/x", "Mul_328"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}, "name": "tf_op_layer_Pow_53", "inbound_nodes": [[["tf_op_layer_Mul_328", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_329", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_329", "op": "Mul", "input": ["dense_431/Identity", "Pow_53"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_329", "inbound_nodes": [[["dense_431", 0, 0, {}], ["tf_op_layer_Pow_53", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_216"}, "name": "input_216", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_49", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_49", "op": "Relu", "input": ["Mul_329"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Relu_49", "inbound_nodes": [[["tf_op_layer_Mul_329", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Max_57", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_57", "op": "Max", "input": ["input_216", "Max_57/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Max_57", "inbound_nodes": [[["input_216", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_330", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_330", "op": "Mul", "input": ["Relu_49", "Max_57"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_330", "inbound_nodes": [[["tf_op_layer_Relu_49", 0, 0, {}], ["tf_op_layer_Max_57", 0, 0, {}]]]}], "input_layers": [["latent_params", 0, 0], ["conditional_params", 0, 0], ["input_216", 0, 0]], "output_layers": [["tf_op_layer_Mul_330", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 6]}, {"class_name": "TensorShape", "items": [null, 32, 1]}, {"class_name": "TensorShape", "items": [null, 32, 288]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_107", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}, "name": "latent_params", "inbound_nodes": []}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector_53", "trainable": true, "dtype": "float32", "n": 32}, "name": "repeat_vector_53", "inbound_nodes": [[["latent_params", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}, "name": "conditional_params", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_432", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_432", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_432/begin", "strided_slice_432/end", "strided_slice_432/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_432", "inbound_nodes": [[["repeat_vector_53", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_435", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_435", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_435/begin", "strided_slice_435/end", "strided_slice_435/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "2"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_435", "inbound_nodes": [[["repeat_vector_53", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_106", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_106", "op": "AddV2", "input": ["conditional_params_56", "strided_slice_432"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_106", "inbound_nodes": [[["conditional_params", 0, 0, {}], ["tf_op_layer_strided_slice_432", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_161", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_161", "inbound_nodes": [[["tf_op_layer_strided_slice_435", 0, 0, {}], ["tf_op_layer_AddV2_106", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_434", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_434", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_434/begin", "strided_slice_434/end", "strided_slice_434/strides"], "attr": {"shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_434", "inbound_nodes": [[["repeat_vector_53", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_428", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_428", "inbound_nodes": [[["concatenate_161", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "color_law", "inbound_nodes": [[["tf_op_layer_strided_slice_434", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_433", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_433", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_433/begin", "strided_slice_433/end", "strided_slice_433/strides"], "attr": {"Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_433", "inbound_nodes": [[["repeat_vector_53", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_429", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_429", "inbound_nodes": [[["dense_428", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_107", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_107", "op": "AddV2", "input": ["color_law_56/Identity", "strided_slice_433"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_107", "inbound_nodes": [[["color_law", 0, 0, {}], ["tf_op_layer_strided_slice_433", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_430", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_430", "inbound_nodes": [[["dense_429", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_328", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_328", "op": "Mul", "input": ["Mul_328/x", "AddV2_107"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}, "name": "tf_op_layer_Mul_328", "inbound_nodes": [[["tf_op_layer_AddV2_107", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_431", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_431", "inbound_nodes": [[["dense_430", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow_53", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_53", "op": "Pow", "input": ["Pow_53/x", "Mul_328"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}, "name": "tf_op_layer_Pow_53", "inbound_nodes": [[["tf_op_layer_Mul_328", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_329", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_329", "op": "Mul", "input": ["dense_431/Identity", "Pow_53"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_329", "inbound_nodes": [[["dense_431", 0, 0, {}], ["tf_op_layer_Pow_53", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_216"}, "name": "input_216", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_49", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_49", "op": "Relu", "input": ["Mul_329"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Relu_49", "inbound_nodes": [[["tf_op_layer_Mul_329", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Max_57", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_57", "op": "Max", "input": ["input_216", "Max_57/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Max_57", "inbound_nodes": [[["input_216", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_330", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_330", "op": "Mul", "input": ["Relu_49", "Max_57"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_330", "inbound_nodes": [[["tf_op_layer_Relu_49", 0, 0, {}], ["tf_op_layer_Max_57", 0, 0, {}]]]}], "input_layers": [["latent_params", 0, 0], ["conditional_params", 0, 0], ["input_216", 0, 0]], "output_layers": [["tf_op_layer_Mul_330", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "latent_params", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}}
?
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "RepeatVector", "name": "repeat_vector_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "repeat_vector_53", "trainable": true, "dtype": "float32", "n": 32}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conditional_params", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}}
?
 	variables
!regularization_losses
"trainable_variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_432", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_432", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_432", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_432/begin", "strided_slice_432/end", "strided_slice_432/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}}
?
$	variables
%regularization_losses
&trainable_variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_435", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_435", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_435", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_435/begin", "strided_slice_435/end", "strided_slice_435/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "2"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}}
?
(	variables
)regularization_losses
*trainable_variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_106", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_106", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_106", "op": "AddV2", "input": ["conditional_params_56", "strided_slice_432"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
,	variables
-regularization_losses
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_161", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_161", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 3]}, {"class_name": "TensorShape", "items": [null, 32, 1]}]}
?
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_434", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_434", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_434", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_434/begin", "strided_slice_434/end", "strided_slice_434/strides"], "attr": {"shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}}
?

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_428", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_428", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 4]}}
?4

:kernel
;	variables
<regularization_losses
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?2
_tf_keras_layer?2{"class_name": "Dense", "name": "color_law", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 1]}}
?
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_433", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_433", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_433", "op": "StridedSlice", "input": ["repeat_vector_53/Identity", "strided_slice_433/begin", "strided_slice_433/end", "strided_slice_433/strides"], "attr": {"Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}}
?

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_429", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_429", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32]}}
?
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_107", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_107", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_107", "op": "AddV2", "input": ["color_law_56/Identity", "strided_slice_433"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?

Mkernel
Nbias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_430", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_430", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 128]}}
?
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_328", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_328", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_328", "op": "Mul", "input": ["Mul_328/x", "AddV2_107"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}}
?

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_431", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_431", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 256]}}
?
]	variables
^regularization_losses
_trainable_variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Pow_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Pow_53", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_53", "op": "Pow", "input": ["Pow_53/x", "Mul_328"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}}
?
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_329", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_329", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_329", "op": "Mul", "input": ["dense_431/Identity", "Pow_53"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_216", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_216"}}
?
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Relu_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Relu_49", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_49", "op": "Relu", "input": ["Mul_329"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Max_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Max_57", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_57", "op": "Max", "input": ["input_216", "Max_57/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}}
?
m	variables
nregularization_losses
otrainable_variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_330", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_330", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_330", "op": "Mul", "input": ["Relu_49", "Max_57"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
_
40
51
:2
C3
D4
M5
N6
W7
X8"
trackable_list_wrapper
 "
trackable_list_wrapper
X
40
51
C2
D3
M4
N5
W6
X7"
trackable_list_wrapper
?
	variables
qlayer_regularization_losses
rnon_trainable_variables
smetrics
tlayer_metrics
regularization_losses

ulayers
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
vlayer_regularization_losses
wnon_trainable_variables
xmetrics
ylayer_metrics
regularization_losses

zlayers
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 	variables
{layer_regularization_losses
|non_trainable_variables
}metrics
~layer_metrics
!regularization_losses

layers
"trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
%regularization_losses
?layers
&trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
(	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
)regularization_losses
?layers
*trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
-regularization_losses
?layers
.trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
1regularization_losses
?layers
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
":  2dense_428/kernel
: 2dense_428/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
6	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
7regularization_losses
?layers
8trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$	?2color_law_56/kernel
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
<regularization_losses
?layers
=trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
@regularization_losses
?layers
Atrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	 ?2dense_429/kernel
:?2dense_429/bias
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
E	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Fregularization_losses
?layers
Gtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
I	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Jregularization_losses
?layers
Ktrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_430/kernel
:?2dense_430/bias
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
O	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Pregularization_losses
?layers
Qtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
S	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Tregularization_losses
?layers
Utrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_431/kernel
:?2dense_431/bias
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
Y	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
Zregularization_losses
?layers
[trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
^regularization_losses
?layers
_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
a	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
bregularization_losses
?layers
ctrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
e	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
fregularization_losses
?layers
gtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
i	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
jregularization_losses
?layers
ktrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
m	variables
 ?layer_regularization_losses
?non_trainable_variables
?metrics
?layer_metrics
nregularization_losses
?layers
otrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
*__inference_model_107_layer_call_fn_446965
*__inference_model_107_layer_call_fn_446550
*__inference_model_107_layer_call_fn_446940
*__inference_model_107_layer_call_fn_446482?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_445926?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_216????????? ?
?2?
E__inference_model_107_layer_call_and_return_conditional_losses_446370
E__inference_model_107_layer_call_and_return_conditional_losses_446746
E__inference_model_107_layer_call_and_return_conditional_losses_446413
E__inference_model_107_layer_call_and_return_conditional_losses_446915?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_repeat_vector_53_layer_call_fn_445941?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!???????????????????
?2?
L__inference_repeat_vector_53_layer_call_and_return_conditional_losses_445935?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!???????????????????
?2?
>__inference_tf_op_layer_strided_slice_432_layer_call_fn_446978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_446973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_tf_op_layer_strided_slice_435_layer_call_fn_446991?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_446986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_tf_op_layer_AddV2_106_layer_call_fn_447003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_446997?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_concatenate_161_layer_call_fn_447016?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_concatenate_161_layer_call_and_return_conditional_losses_447010?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_tf_op_layer_strided_slice_434_layer_call_fn_447029?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_447024?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_428_layer_call_fn_447069?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_428_layer_call_and_return_conditional_losses_447060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_color_law_layer_call_fn_447103?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_color_law_layer_call_and_return_conditional_losses_447096?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_tf_op_layer_strided_slice_433_layer_call_fn_447116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_447111?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_429_layer_call_fn_447156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_429_layer_call_and_return_conditional_losses_447147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_tf_op_layer_AddV2_107_layer_call_fn_447168?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_447162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_430_layer_call_fn_447208?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_430_layer_call_and_return_conditional_losses_447199?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Mul_328_layer_call_fn_447219?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_447214?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_431_layer_call_fn_447258?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_431_layer_call_and_return_conditional_losses_447249?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_tf_op_layer_Pow_53_layer_call_fn_447269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_447264?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Mul_329_layer_call_fn_447281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_447275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Relu_49_layer_call_fn_447291?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_447286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_tf_op_layer_Max_57_layer_call_fn_447302?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_447297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Mul_330_layer_call_fn_447314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_447308?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
VBT
$__inference_signature_wrapper_446577conditional_params	input_216latent_params?
!__inference__wrapped_model_445926?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_216????????? ?
? "N?K
I
tf_op_layer_Mul_3302?/
tf_op_layer_Mul_330????????? ??
E__inference_color_law_layer_call_and_return_conditional_losses_447096d:3?0
)?&
$?!
inputs????????? 
? "*?'
 ?
0????????? ?
? ?
*__inference_color_law_layer_call_fn_447103W:3?0
)?&
$?!
inputs????????? 
? "?????????? ??
K__inference_concatenate_161_layer_call_and_return_conditional_losses_447010?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? ")?&
?
0????????? 
? ?
0__inference_concatenate_161_layer_call_fn_447016?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? "?????????? ?
E__inference_dense_428_layer_call_and_return_conditional_losses_447060d453?0
)?&
$?!
inputs????????? 
? ")?&
?
0?????????  
? ?
*__inference_dense_428_layer_call_fn_447069W453?0
)?&
$?!
inputs????????? 
? "??????????  ?
E__inference_dense_429_layer_call_and_return_conditional_losses_447147eCD3?0
)?&
$?!
inputs?????????  
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_429_layer_call_fn_447156XCD3?0
)?&
$?!
inputs?????????  
? "?????????? ??
E__inference_dense_430_layer_call_and_return_conditional_losses_447199fMN4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_430_layer_call_fn_447208YMN4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
E__inference_dense_431_layer_call_and_return_conditional_losses_447249fWX4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_431_layer_call_fn_447258YWX4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
E__inference_model_107_layer_call_and_return_conditional_losses_446370?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_216????????? ?
p

 
? "*?'
 ?
0????????? ?
? ?
E__inference_model_107_layer_call_and_return_conditional_losses_446413?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_216????????? ?
p 

 
? "*?'
 ?
0????????? ?
? ?
E__inference_model_107_layer_call_and_return_conditional_losses_446746?	:45CDMNWX???
???
x?u
"?
inputs/0?????????
&?#
inputs/1????????? 
'?$
inputs/2????????? ?
p

 
? "*?'
 ?
0????????? ?
? ?
E__inference_model_107_layer_call_and_return_conditional_losses_446915?	:45CDMNWX???
???
x?u
"?
inputs/0?????????
&?#
inputs/1????????? 
'?$
inputs/2????????? ?
p 

 
? "*?'
 ?
0????????? ?
? ?
*__inference_model_107_layer_call_fn_446482?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_216????????? ?
p

 
? "?????????? ??
*__inference_model_107_layer_call_fn_446550?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_216????????? ?
p 

 
? "?????????? ??
*__inference_model_107_layer_call_fn_446940?	:45CDMNWX???
???
x?u
"?
inputs/0?????????
&?#
inputs/1????????? 
'?$
inputs/2????????? ?
p

 
? "?????????? ??
*__inference_model_107_layer_call_fn_446965?	:45CDMNWX???
???
x?u
"?
inputs/0?????????
&?#
inputs/1????????? 
'?$
inputs/2????????? ?
p 

 
? "?????????? ??
L__inference_repeat_vector_53_layer_call_and_return_conditional_losses_445935n8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0????????? ?????????
? ?
1__inference_repeat_vector_53_layer_call_fn_445941a8?5
.?+
)?&
inputs??????????????????
? "%?"????????? ??????????
$__inference_signature_wrapper_446577?	:45CDMNWX???
? 
???
F
conditional_params0?-
conditional_params????????? 
5
	input_216(?%
	input_216????????? ?
8
latent_params'?$
latent_params?????????"N?K
I
tf_op_layer_Mul_3302?/
tf_op_layer_Mul_330????????? ??
Q__inference_tf_op_layer_AddV2_106_layer_call_and_return_conditional_losses_446997?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? ")?&
?
0????????? 
? ?
6__inference_tf_op_layer_AddV2_106_layer_call_fn_447003?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? "?????????? ?
Q__inference_tf_op_layer_AddV2_107_layer_call_and_return_conditional_losses_447162?c?`
Y?V
T?Q
'?$
inputs/0????????? ?
&?#
inputs/1????????? 
? "*?'
 ?
0????????? ?
? ?
6__inference_tf_op_layer_AddV2_107_layer_call_fn_447168?c?`
Y?V
T?Q
'?$
inputs/0????????? ?
&?#
inputs/1????????? 
? "?????????? ??
N__inference_tf_op_layer_Max_57_layer_call_and_return_conditional_losses_447297a4?1
*?'
%?"
inputs????????? ?
? ")?&
?
0????????? 
? ?
3__inference_tf_op_layer_Max_57_layer_call_fn_447302T4?1
*?'
%?"
inputs????????? ?
? "?????????? ?
O__inference_tf_op_layer_Mul_328_layer_call_and_return_conditional_losses_447214b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
4__inference_tf_op_layer_Mul_328_layer_call_fn_447219U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
O__inference_tf_op_layer_Mul_329_layer_call_and_return_conditional_losses_447275?d?a
Z?W
U?R
'?$
inputs/0????????? ?
'?$
inputs/1????????? ?
? "*?'
 ?
0????????? ?
? ?
4__inference_tf_op_layer_Mul_329_layer_call_fn_447281?d?a
Z?W
U?R
'?$
inputs/0????????? ?
'?$
inputs/1????????? ?
? "?????????? ??
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_447308?c?`
Y?V
T?Q
'?$
inputs/0????????? ?
&?#
inputs/1????????? 
? "*?'
 ?
0????????? ?
? ?
4__inference_tf_op_layer_Mul_330_layer_call_fn_447314?c?`
Y?V
T?Q
'?$
inputs/0????????? ?
&?#
inputs/1????????? 
? "?????????? ??
N__inference_tf_op_layer_Pow_53_layer_call_and_return_conditional_losses_447264b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
3__inference_tf_op_layer_Pow_53_layer_call_fn_447269U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
O__inference_tf_op_layer_Relu_49_layer_call_and_return_conditional_losses_447286b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
4__inference_tf_op_layer_Relu_49_layer_call_fn_447291U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
Y__inference_tf_op_layer_strided_slice_432_layer_call_and_return_conditional_losses_446973`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_432_layer_call_fn_446978S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_433_layer_call_and_return_conditional_losses_447111`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_433_layer_call_fn_447116S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_434_layer_call_and_return_conditional_losses_447024`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_434_layer_call_fn_447029S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_435_layer_call_and_return_conditional_losses_446986`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_435_layer_call_fn_446991S3?0
)?&
$?!
inputs????????? 
? "?????????? 