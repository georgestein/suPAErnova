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
dense_460/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_460/kernel
u
$dense_460/kernel/Read/ReadVariableOpReadVariableOpdense_460/kernel*
_output_shapes

: *
dtype0
t
dense_460/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_460/bias
m
"dense_460/bias/Read/ReadVariableOpReadVariableOpdense_460/bias*
_output_shapes
: *
dtype0
?
color_law_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_namecolor_law_60/kernel
|
'color_law_60/kernel/Read/ReadVariableOpReadVariableOpcolor_law_60/kernel*
_output_shapes
:	?*
dtype0
}
dense_461/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*!
shared_namedense_461/kernel
v
$dense_461/kernel/Read/ReadVariableOpReadVariableOpdense_461/kernel*
_output_shapes
:	 ?*
dtype0
u
dense_461/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_461/bias
n
"dense_461/bias/Read/ReadVariableOpReadVariableOpdense_461/bias*
_output_shapes	
:?*
dtype0
~
dense_462/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_462/kernel
w
$dense_462/kernel/Read/ReadVariableOpReadVariableOpdense_462/kernel* 
_output_shapes
:
??*
dtype0
u
dense_462/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_462/bias
n
"dense_462/bias/Read/ReadVariableOpReadVariableOpdense_462/bias*
_output_shapes	
:?*
dtype0
~
dense_463/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_463/kernel
w
$dense_463/kernel/Read/ReadVariableOpReadVariableOpdense_463/kernel* 
_output_shapes
:
??*
dtype0
u
dense_463/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_463/bias
n
"dense_463/bias/Read/ReadVariableOpReadVariableOpdense_463/bias*
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
 
R
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
R
(regularization_losses
)	variables
*trainable_variables
+	keras_api
R
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
^

:kernel
;regularization_losses
<	variables
=trainable_variables
>	keras_api
R
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
R
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
h

Mkernel
Nbias
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
R
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
h

Wkernel
Xbias
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
R
]regularization_losses
^	variables
_trainable_variables
`	keras_api
R
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
 
R
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
R
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
R
mregularization_losses
n	variables
otrainable_variables
p	keras_api
 
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
regularization_losses
qnon_trainable_variables
	variables
trainable_variables
rlayer_metrics
smetrics
tlayer_regularization_losses

ulayers
 
 
 
 
?
vlayer_regularization_losses
regularization_losses
wnon_trainable_variables
	variables
trainable_variables
xlayer_metrics
ymetrics

zlayers
 
 
 
?
{layer_regularization_losses
 regularization_losses
|non_trainable_variables
!	variables
"trainable_variables
}layer_metrics
~metrics

layers
 
 
 
?
 ?layer_regularization_losses
$regularization_losses
?non_trainable_variables
%	variables
&trainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
(regularization_losses
?non_trainable_variables
)	variables
*trainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
,regularization_losses
?non_trainable_variables
-	variables
.trainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
0regularization_losses
?non_trainable_variables
1	variables
2trainable_variables
?layer_metrics
?metrics
?layers
\Z
VARIABLE_VALUEdense_460/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_460/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
?
 ?layer_regularization_losses
6regularization_losses
?non_trainable_variables
7	variables
8trainable_variables
?layer_metrics
?metrics
?layers
_]
VARIABLE_VALUEcolor_law_60/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

:0
 
?
 ?layer_regularization_losses
;regularization_losses
?non_trainable_variables
<	variables
=trainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?regularization_losses
?non_trainable_variables
@	variables
Atrainable_variables
?layer_metrics
?metrics
?layers
\Z
VARIABLE_VALUEdense_461/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_461/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
?
 ?layer_regularization_losses
Eregularization_losses
?non_trainable_variables
F	variables
Gtrainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
Iregularization_losses
?non_trainable_variables
J	variables
Ktrainable_variables
?layer_metrics
?metrics
?layers
\Z
VARIABLE_VALUEdense_462/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_462/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1

M0
N1
?
 ?layer_regularization_losses
Oregularization_losses
?non_trainable_variables
P	variables
Qtrainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
Sregularization_losses
?non_trainable_variables
T	variables
Utrainable_variables
?layer_metrics
?metrics
?layers
\Z
VARIABLE_VALUEdense_463/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_463/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

W0
X1

W0
X1
?
 ?layer_regularization_losses
Yregularization_losses
?non_trainable_variables
Z	variables
[trainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
]regularization_losses
?non_trainable_variables
^	variables
_trainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
aregularization_losses
?non_trainable_variables
b	variables
ctrainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
eregularization_losses
?non_trainable_variables
f	variables
gtrainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
iregularization_losses
?non_trainable_variables
j	variables
ktrainable_variables
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
mregularization_losses
?non_trainable_variables
n	variables
otrainable_variables
?layer_metrics
?metrics
?layers

:0
 
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
serving_default_input_232Placeholder*,
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
StatefulPartitionedCallStatefulPartitionedCall"serving_default_conditional_paramsserving_default_input_232serving_default_latent_paramscolor_law_60/kerneldense_460/kerneldense_460/biasdense_461/kerneldense_461/biasdense_462/kerneldense_462/biasdense_463/kerneldense_463/bias*
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
$__inference_signature_wrapper_457327
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_460/kernel/Read/ReadVariableOp"dense_460/bias/Read/ReadVariableOp'color_law_60/kernel/Read/ReadVariableOp$dense_461/kernel/Read/ReadVariableOp"dense_461/bias/Read/ReadVariableOp$dense_462/kernel/Read/ReadVariableOp"dense_462/bias/Read/ReadVariableOp$dense_463/kernel/Read/ReadVariableOp"dense_463/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_458120
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_460/kerneldense_460/biascolor_law_60/kerneldense_461/kerneldense_461/biasdense_462/kerneldense_462/biasdense_463/kerneldense_463/bias*
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
"__inference__traced_restore_458159??
?
Z
>__inference_tf_op_layer_strided_slice_466_layer_call_fn_457779

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
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_4567532
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
?
u
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_456753

inputs
identity?
strided_slice_466/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_466/begin
strided_slice_466/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_466/end?
strided_slice_466/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_466/strides?
strided_slice_466StridedSliceinputs strided_slice_466/begin:output:0strided_slice_466/end:output:0"strided_slice_466/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_466r
IdentityIdentitystrided_slice_466:output:0*
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
j
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_458014

inputs
identityY
Pow_57/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

Pow_57/xx
Pow_57PowPow_57/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2
Pow_57c
IdentityIdentity
Pow_57:z:0*
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
?
?
E__inference_dense_462_layer_call_and_return_conditional_losses_456986

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
?
{
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_458025
inputs_0
inputs_1
identitys
Mul_357Mulinputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_357d
IdentityIdentityMul_357:z:0*
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
?K
?
E__inference_model_115_layer_call_and_return_conditional_losses_457163
latent_params
conditional_params
	input_232
color_law_457131
dense_460_457135
dense_460_457137
dense_461_457141
dense_461_457143
dense_462_457147
dense_462_457149
dense_463_457152
dense_463_457154
identity??!color_law/StatefulPartitionedCall?!dense_460/StatefulPartitionedCall?!dense_461/StatefulPartitionedCall?!dense_462/StatefulPartitionedCall?!dense_463/StatefulPartitionedCall?
 repeat_vector_57/PartitionedCallPartitionedCalllatent_params*
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
L__inference_repeat_vector_57_layer_call_and_return_conditional_losses_4566852"
 repeat_vector_57/PartitionedCall?
-tf_op_layer_strided_slice_464/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_4567062/
-tf_op_layer_strided_slice_464/PartitionedCall?
-tf_op_layer_strided_slice_467/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_4567222/
-tf_op_layer_strided_slice_467/PartitionedCall?
%tf_op_layer_AddV2_114/PartitionedCallPartitionedCallconditional_params6tf_op_layer_strided_slice_464/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_4567362'
%tf_op_layer_AddV2_114/PartitionedCall?
-tf_op_layer_strided_slice_466/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_4567532/
-tf_op_layer_strided_slice_466/PartitionedCall?
concatenate_173/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_467/PartitionedCall:output:0.tf_op_layer_AddV2_114/PartitionedCall:output:0*
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
K__inference_concatenate_173_layer_call_and_return_conditional_losses_4567682!
concatenate_173/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_466/PartitionedCall:output:0color_law_457131*
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
E__inference_color_law_layer_call_and_return_conditional_losses_4568042#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_465/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_4568242/
-tf_op_layer_strided_slice_465/PartitionedCall?
!dense_460/StatefulPartitionedCallStatefulPartitionedCall(concatenate_173/PartitionedCall:output:0dense_460_457135dense_460_457137*
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
E__inference_dense_460_layer_call_and_return_conditional_losses_4568632#
!dense_460/StatefulPartitionedCall?
%tf_op_layer_AddV2_115/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_465/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_4568852'
%tf_op_layer_AddV2_115/PartitionedCall?
!dense_461/StatefulPartitionedCallStatefulPartitionedCall*dense_460/StatefulPartitionedCall:output:0dense_461_457141dense_461_457143*
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
E__inference_dense_461_layer_call_and_return_conditional_losses_4569252#
!dense_461/StatefulPartitionedCall?
#tf_op_layer_Mul_356/PartitionedCallPartitionedCall.tf_op_layer_AddV2_115/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4569472%
#tf_op_layer_Mul_356/PartitionedCall?
!dense_462/StatefulPartitionedCallStatefulPartitionedCall*dense_461/StatefulPartitionedCall:output:0dense_462_457147dense_462_457149*
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
E__inference_dense_462_layer_call_and_return_conditional_losses_4569862#
!dense_462/StatefulPartitionedCall?
!dense_463/StatefulPartitionedCallStatefulPartitionedCall*dense_462/StatefulPartitionedCall:output:0dense_463_457152dense_463_457154*
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
E__inference_dense_463_layer_call_and_return_conditional_losses_4570322#
!dense_463/StatefulPartitionedCall?
"tf_op_layer_Pow_57/PartitionedCallPartitionedCall,tf_op_layer_Mul_356/PartitionedCall:output:0*
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
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_4570542$
"tf_op_layer_Pow_57/PartitionedCall?
#tf_op_layer_Mul_357/PartitionedCallPartitionedCall*dense_463/StatefulPartitionedCall:output:0+tf_op_layer_Pow_57/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_4570682%
#tf_op_layer_Mul_357/PartitionedCall?
#tf_op_layer_Relu_53/PartitionedCallPartitionedCall,tf_op_layer_Mul_357/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_4570822%
#tf_op_layer_Relu_53/PartitionedCall?
"tf_op_layer_Max_61/PartitionedCallPartitionedCall	input_232*
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
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_4570962$
"tf_op_layer_Max_61/PartitionedCall?
#tf_op_layer_Mul_358/PartitionedCallPartitionedCall,tf_op_layer_Relu_53/PartitionedCall:output:0+tf_op_layer_Max_61/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_4571102%
#tf_op_layer_Mul_358/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_358/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_460/StatefulPartitionedCall"^dense_461/StatefulPartitionedCall"^dense_462/StatefulPartitionedCall"^dense_463/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_460/StatefulPartitionedCall!dense_460/StatefulPartitionedCall2F
!dense_461/StatefulPartitionedCall!dense_461/StatefulPartitionedCall2F
!dense_462/StatefulPartitionedCall!dense_462/StatefulPartitionedCall2F
!dense_463/StatefulPartitionedCall!dense_463/StatefulPartitionedCall:V R
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
_user_specified_name	input_232:
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
?
E__inference_model_115_layer_call_and_return_conditional_losses_457279

inputs
inputs_1
inputs_2
color_law_457247
dense_460_457251
dense_460_457253
dense_461_457257
dense_461_457259
dense_462_457263
dense_462_457265
dense_463_457268
dense_463_457270
identity??!color_law/StatefulPartitionedCall?!dense_460/StatefulPartitionedCall?!dense_461/StatefulPartitionedCall?!dense_462/StatefulPartitionedCall?!dense_463/StatefulPartitionedCall?
 repeat_vector_57/PartitionedCallPartitionedCallinputs*
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
L__inference_repeat_vector_57_layer_call_and_return_conditional_losses_4566852"
 repeat_vector_57/PartitionedCall?
-tf_op_layer_strided_slice_464/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_4567062/
-tf_op_layer_strided_slice_464/PartitionedCall?
-tf_op_layer_strided_slice_467/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_4567222/
-tf_op_layer_strided_slice_467/PartitionedCall?
%tf_op_layer_AddV2_114/PartitionedCallPartitionedCallinputs_16tf_op_layer_strided_slice_464/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_4567362'
%tf_op_layer_AddV2_114/PartitionedCall?
-tf_op_layer_strided_slice_466/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_4567532/
-tf_op_layer_strided_slice_466/PartitionedCall?
concatenate_173/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_467/PartitionedCall:output:0.tf_op_layer_AddV2_114/PartitionedCall:output:0*
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
K__inference_concatenate_173_layer_call_and_return_conditional_losses_4567682!
concatenate_173/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_466/PartitionedCall:output:0color_law_457247*
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
E__inference_color_law_layer_call_and_return_conditional_losses_4568042#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_465/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_4568242/
-tf_op_layer_strided_slice_465/PartitionedCall?
!dense_460/StatefulPartitionedCallStatefulPartitionedCall(concatenate_173/PartitionedCall:output:0dense_460_457251dense_460_457253*
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
E__inference_dense_460_layer_call_and_return_conditional_losses_4568632#
!dense_460/StatefulPartitionedCall?
%tf_op_layer_AddV2_115/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_465/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_4568852'
%tf_op_layer_AddV2_115/PartitionedCall?
!dense_461/StatefulPartitionedCallStatefulPartitionedCall*dense_460/StatefulPartitionedCall:output:0dense_461_457257dense_461_457259*
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
E__inference_dense_461_layer_call_and_return_conditional_losses_4569252#
!dense_461/StatefulPartitionedCall?
#tf_op_layer_Mul_356/PartitionedCallPartitionedCall.tf_op_layer_AddV2_115/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4569472%
#tf_op_layer_Mul_356/PartitionedCall?
!dense_462/StatefulPartitionedCallStatefulPartitionedCall*dense_461/StatefulPartitionedCall:output:0dense_462_457263dense_462_457265*
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
E__inference_dense_462_layer_call_and_return_conditional_losses_4569862#
!dense_462/StatefulPartitionedCall?
!dense_463/StatefulPartitionedCallStatefulPartitionedCall*dense_462/StatefulPartitionedCall:output:0dense_463_457268dense_463_457270*
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
E__inference_dense_463_layer_call_and_return_conditional_losses_4570322#
!dense_463/StatefulPartitionedCall?
"tf_op_layer_Pow_57/PartitionedCallPartitionedCall,tf_op_layer_Mul_356/PartitionedCall:output:0*
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
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_4570542$
"tf_op_layer_Pow_57/PartitionedCall?
#tf_op_layer_Mul_357/PartitionedCallPartitionedCall*dense_463/StatefulPartitionedCall:output:0+tf_op_layer_Pow_57/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_4570682%
#tf_op_layer_Mul_357/PartitionedCall?
#tf_op_layer_Relu_53/PartitionedCallPartitionedCall,tf_op_layer_Mul_357/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_4570822%
#tf_op_layer_Relu_53/PartitionedCall?
"tf_op_layer_Max_61/PartitionedCallPartitionedCallinputs_2*
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
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_4570962$
"tf_op_layer_Max_61/PartitionedCall?
#tf_op_layer_Mul_358/PartitionedCallPartitionedCall,tf_op_layer_Relu_53/PartitionedCall:output:0+tf_op_layer_Max_61/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_4571102%
#tf_op_layer_Mul_358/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_358/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_460/StatefulPartitionedCall"^dense_461/StatefulPartitionedCall"^dense_462/StatefulPartitionedCall"^dense_463/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_460/StatefulPartitionedCall!dense_460/StatefulPartitionedCall2F
!dense_461/StatefulPartitionedCall!dense_461/StatefulPartitionedCall2F
!dense_462/StatefulPartitionedCall!dense_462/StatefulPartitionedCall2F
!dense_463/StatefulPartitionedCall!dense_463/StatefulPartitionedCall:O K
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
?

*__inference_dense_460_layer_call_fn_457819

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
E__inference_dense_460_layer_call_and_return_conditional_losses_4568632
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
?
P
4__inference_tf_op_layer_Mul_356_layer_call_fn_457969

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
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4569472
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
?.
?
"__inference__traced_restore_458159
file_prefix%
!assignvariableop_dense_460_kernel%
!assignvariableop_1_dense_460_bias*
&assignvariableop_2_color_law_60_kernel'
#assignvariableop_3_dense_461_kernel%
!assignvariableop_4_dense_461_bias'
#assignvariableop_5_dense_462_kernel%
!assignvariableop_6_dense_462_bias'
#assignvariableop_7_dense_463_kernel%
!assignvariableop_8_dense_463_bias
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_460_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_460_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_color_law_60_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_461_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_461_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_462_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_462_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_463_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_463_biasIdentity_8:output:0*
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
?
b
6__inference_tf_op_layer_AddV2_115_layer_call_fn_457918
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
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_4568852
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
>__inference_tf_op_layer_strided_slice_465_layer_call_fn_457866

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
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_4568242
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
E__inference_dense_463_layer_call_and_return_conditional_losses_457032

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
?
Z
>__inference_tf_op_layer_strided_slice_467_layer_call_fn_457741

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
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_4567222
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
?
w
K__inference_concatenate_173_layer_call_and_return_conditional_losses_457760
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
?
u
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_456722

inputs
identity?
strided_slice_467/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_467/begin
strided_slice_467/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_467/end?
strided_slice_467/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_467/strides?
strided_slice_467StridedSliceinputs strided_slice_467/begin:output:0strided_slice_467/end:output:0"strided_slice_467/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2
strided_slice_467r
IdentityIdentitystrided_slice_467:output:0*
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
?
?
$__inference_signature_wrapper_457327
conditional_params
	input_232
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
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_232unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
!__inference__wrapped_model_4566762
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
_user_specified_name	input_232:VR
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
?
O
3__inference_tf_op_layer_Pow_57_layer_call_fn_458019

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
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_4570542
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
?
{
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_456736

inputs
inputs_1
identityv
	AddV2_114AddV2inputsinputs_1*
T0*
_cloned(*+
_output_shapes
:????????? 2
	AddV2_114e
IdentityIdentityAddV2_114:z:0*
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
u
K__inference_concatenate_173_layer_call_and_return_conditional_losses_456768

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
?
?
E__inference_dense_460_layer_call_and_return_conditional_losses_456863

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
?

*__inference_dense_462_layer_call_fn_457958

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
E__inference_dense_462_layer_call_and_return_conditional_losses_4569862
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
?
y
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_457110

inputs
inputs_1
identityq
Mul_358Mulinputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_358d
IdentityIdentityMul_358:z:0*
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
?
M
1__inference_repeat_vector_57_layer_call_fn_456691

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
L__inference_repeat_vector_57_layer_call_and_return_conditional_losses_4566852
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
?
?
E__inference_dense_463_layer_call_and_return_conditional_losses_457999

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
?
k
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_458036

inputs
identityh
Relu_53Reluinputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Relu_53n
IdentityIdentityRelu_53:activations:0*
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
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_457861

inputs
identity?
strided_slice_465/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_465/begin
strided_slice_465/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_465/end?
strided_slice_465/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_465/strides?
strided_slice_465StridedSliceinputs strided_slice_465/begin:output:0strided_slice_465/end:output:0"strided_slice_465/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_465r
IdentityIdentitystrided_slice_465:output:0*
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
p
*__inference_color_law_layer_call_fn_457853

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
E__inference_color_law_layer_call_and_return_conditional_losses_4568042
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
?
?
*__inference_model_115_layer_call_fn_457715
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
E__inference_model_115_layer_call_and_return_conditional_losses_4572792
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
?
O
3__inference_tf_op_layer_Max_61_layer_call_fn_458052

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
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_4570962
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
?
`
4__inference_tf_op_layer_Mul_358_layer_call_fn_458064
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
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_4571102
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
??
?
E__inference_model_115_layer_call_and_return_conditional_losses_457496
inputs_0
inputs_1
inputs_2/
+color_law_tensordot_readvariableop_resource/
+dense_460_tensordot_readvariableop_resource-
)dense_460_biasadd_readvariableop_resource/
+dense_461_tensordot_readvariableop_resource-
)dense_461_biasadd_readvariableop_resource/
+dense_462_tensordot_readvariableop_resource-
)dense_462_biasadd_readvariableop_resource/
+dense_463_tensordot_readvariableop_resource-
)dense_463_biasadd_readvariableop_resource
identity??
repeat_vector_57/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_57/ExpandDims/dim?
repeat_vector_57/ExpandDims
ExpandDimsinputs_0(repeat_vector_57/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
repeat_vector_57/ExpandDims?
repeat_vector_57/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
repeat_vector_57/stack?
repeat_vector_57/TileTile$repeat_vector_57/ExpandDims:output:0repeat_vector_57/stack:output:0*
T0*+
_output_shapes
:????????? 2
repeat_vector_57/Tile?
5tf_op_layer_strided_slice_464/strided_slice_464/beginConst*
_output_shapes
:*
dtype0*
valueB"        27
5tf_op_layer_strided_slice_464/strided_slice_464/begin?
3tf_op_layer_strided_slice_464/strided_slice_464/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_464/strided_slice_464/end?
7tf_op_layer_strided_slice_464/strided_slice_464/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_464/strided_slice_464/strides?
/tf_op_layer_strided_slice_464/strided_slice_464StridedSlicerepeat_vector_57/Tile:output:0>tf_op_layer_strided_slice_464/strided_slice_464/begin:output:0<tf_op_layer_strided_slice_464/strided_slice_464/end:output:0@tf_op_layer_strided_slice_464/strided_slice_464/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_464/strided_slice_464?
5tf_op_layer_strided_slice_467/strided_slice_467/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_467/strided_slice_467/begin?
3tf_op_layer_strided_slice_467/strided_slice_467/endConst*
_output_shapes
:*
dtype0*
valueB"        25
3tf_op_layer_strided_slice_467/strided_slice_467/end?
7tf_op_layer_strided_slice_467/strided_slice_467/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_467/strided_slice_467/strides?
/tf_op_layer_strided_slice_467/strided_slice_467StridedSlicerepeat_vector_57/Tile:output:0>tf_op_layer_strided_slice_467/strided_slice_467/begin:output:0<tf_op_layer_strided_slice_467/strided_slice_467/end:output:0@tf_op_layer_strided_slice_467/strided_slice_467/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask21
/tf_op_layer_strided_slice_467/strided_slice_467?
tf_op_layer_AddV2_114/AddV2_114AddV2inputs_18tf_op_layer_strided_slice_464/strided_slice_464:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2!
tf_op_layer_AddV2_114/AddV2_114?
5tf_op_layer_strided_slice_466/strided_slice_466/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_466/strided_slice_466/begin?
3tf_op_layer_strided_slice_466/strided_slice_466/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_466/strided_slice_466/end?
7tf_op_layer_strided_slice_466/strided_slice_466/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_466/strided_slice_466/strides?
/tf_op_layer_strided_slice_466/strided_slice_466StridedSlicerepeat_vector_57/Tile:output:0>tf_op_layer_strided_slice_466/strided_slice_466/begin:output:0<tf_op_layer_strided_slice_466/strided_slice_466/end:output:0@tf_op_layer_strided_slice_466/strided_slice_466/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_466/strided_slice_466|
concatenate_173/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_173/concat/axis?
concatenate_173/concatConcatV28tf_op_layer_strided_slice_467/strided_slice_467:output:0#tf_op_layer_AddV2_114/AddV2_114:z:0$concatenate_173/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concatenate_173/concat?
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
color_law/Tensordot/ShapeShape8tf_op_layer_strided_slice_466/strided_slice_466:output:0*
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
color_law/Tensordot/transpose	Transpose8tf_op_layer_strided_slice_466/strided_slice_466:output:0#color_law/Tensordot/concat:output:0*
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
5tf_op_layer_strided_slice_465/strided_slice_465/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_465/strided_slice_465/begin?
3tf_op_layer_strided_slice_465/strided_slice_465/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_465/strided_slice_465/end?
7tf_op_layer_strided_slice_465/strided_slice_465/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_465/strided_slice_465/strides?
/tf_op_layer_strided_slice_465/strided_slice_465StridedSlicerepeat_vector_57/Tile:output:0>tf_op_layer_strided_slice_465/strided_slice_465/begin:output:0<tf_op_layer_strided_slice_465/strided_slice_465/end:output:0@tf_op_layer_strided_slice_465/strided_slice_465/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_465/strided_slice_465?
"dense_460/Tensordot/ReadVariableOpReadVariableOp+dense_460_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02$
"dense_460/Tensordot/ReadVariableOp~
dense_460/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_460/Tensordot/axes?
dense_460/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_460/Tensordot/free?
dense_460/Tensordot/ShapeShapeconcatenate_173/concat:output:0*
T0*
_output_shapes
:2
dense_460/Tensordot/Shape?
!dense_460/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_460/Tensordot/GatherV2/axis?
dense_460/Tensordot/GatherV2GatherV2"dense_460/Tensordot/Shape:output:0!dense_460/Tensordot/free:output:0*dense_460/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_460/Tensordot/GatherV2?
#dense_460/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_460/Tensordot/GatherV2_1/axis?
dense_460/Tensordot/GatherV2_1GatherV2"dense_460/Tensordot/Shape:output:0!dense_460/Tensordot/axes:output:0,dense_460/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_460/Tensordot/GatherV2_1?
dense_460/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_460/Tensordot/Const?
dense_460/Tensordot/ProdProd%dense_460/Tensordot/GatherV2:output:0"dense_460/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_460/Tensordot/Prod?
dense_460/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_460/Tensordot/Const_1?
dense_460/Tensordot/Prod_1Prod'dense_460/Tensordot/GatherV2_1:output:0$dense_460/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_460/Tensordot/Prod_1?
dense_460/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_460/Tensordot/concat/axis?
dense_460/Tensordot/concatConcatV2!dense_460/Tensordot/free:output:0!dense_460/Tensordot/axes:output:0(dense_460/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_460/Tensordot/concat?
dense_460/Tensordot/stackPack!dense_460/Tensordot/Prod:output:0#dense_460/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_460/Tensordot/stack?
dense_460/Tensordot/transpose	Transposeconcatenate_173/concat:output:0#dense_460/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_460/Tensordot/transpose?
dense_460/Tensordot/ReshapeReshape!dense_460/Tensordot/transpose:y:0"dense_460/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_460/Tensordot/Reshape?
dense_460/Tensordot/MatMulMatMul$dense_460/Tensordot/Reshape:output:0*dense_460/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_460/Tensordot/MatMul?
dense_460/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_460/Tensordot/Const_2?
!dense_460/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_460/Tensordot/concat_1/axis?
dense_460/Tensordot/concat_1ConcatV2%dense_460/Tensordot/GatherV2:output:0$dense_460/Tensordot/Const_2:output:0*dense_460/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_460/Tensordot/concat_1?
dense_460/TensordotReshape$dense_460/Tensordot/MatMul:product:0%dense_460/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
dense_460/Tensordot?
 dense_460/BiasAdd/ReadVariableOpReadVariableOp)dense_460_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_460/BiasAdd/ReadVariableOp?
dense_460/BiasAddAdddense_460/Tensordot:output:0(dense_460/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
dense_460/BiasAddu
dense_460/ReluReludense_460/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
dense_460/Relu?
tf_op_layer_AddV2_115/AddV2_115AddV2color_law/Tensordot:output:08tf_op_layer_strided_slice_465/strided_slice_465:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2!
tf_op_layer_AddV2_115/AddV2_115?
"dense_461/Tensordot/ReadVariableOpReadVariableOp+dense_461_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"dense_461/Tensordot/ReadVariableOp~
dense_461/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_461/Tensordot/axes?
dense_461/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_461/Tensordot/free?
dense_461/Tensordot/ShapeShapedense_460/Relu:activations:0*
T0*
_output_shapes
:2
dense_461/Tensordot/Shape?
!dense_461/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_461/Tensordot/GatherV2/axis?
dense_461/Tensordot/GatherV2GatherV2"dense_461/Tensordot/Shape:output:0!dense_461/Tensordot/free:output:0*dense_461/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_461/Tensordot/GatherV2?
#dense_461/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_461/Tensordot/GatherV2_1/axis?
dense_461/Tensordot/GatherV2_1GatherV2"dense_461/Tensordot/Shape:output:0!dense_461/Tensordot/axes:output:0,dense_461/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_461/Tensordot/GatherV2_1?
dense_461/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_461/Tensordot/Const?
dense_461/Tensordot/ProdProd%dense_461/Tensordot/GatherV2:output:0"dense_461/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_461/Tensordot/Prod?
dense_461/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_461/Tensordot/Const_1?
dense_461/Tensordot/Prod_1Prod'dense_461/Tensordot/GatherV2_1:output:0$dense_461/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_461/Tensordot/Prod_1?
dense_461/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_461/Tensordot/concat/axis?
dense_461/Tensordot/concatConcatV2!dense_461/Tensordot/free:output:0!dense_461/Tensordot/axes:output:0(dense_461/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_461/Tensordot/concat?
dense_461/Tensordot/stackPack!dense_461/Tensordot/Prod:output:0#dense_461/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_461/Tensordot/stack?
dense_461/Tensordot/transpose	Transposedense_460/Relu:activations:0#dense_461/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2
dense_461/Tensordot/transpose?
dense_461/Tensordot/ReshapeReshape!dense_461/Tensordot/transpose:y:0"dense_461/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_461/Tensordot/Reshape?
dense_461/Tensordot/MatMulMatMul$dense_461/Tensordot/Reshape:output:0*dense_461/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_461/Tensordot/MatMul?
dense_461/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_461/Tensordot/Const_2?
!dense_461/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_461/Tensordot/concat_1/axis?
dense_461/Tensordot/concat_1ConcatV2%dense_461/Tensordot/GatherV2:output:0$dense_461/Tensordot/Const_2:output:0*dense_461/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_461/Tensordot/concat_1?
dense_461/TensordotReshape$dense_461/Tensordot/MatMul:product:0%dense_461/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_461/Tensordot?
 dense_461/BiasAdd/ReadVariableOpReadVariableOp)dense_461_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_461/BiasAdd/ReadVariableOp?
dense_461/BiasAddAdddense_461/Tensordot:output:0(dense_461/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_461/BiasAddv
dense_461/ReluReludense_461/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_461/Relu?
tf_op_layer_Mul_356/Mul_356/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
tf_op_layer_Mul_356/Mul_356/x?
tf_op_layer_Mul_356/Mul_356Mul&tf_op_layer_Mul_356/Mul_356/x:output:0#tf_op_layer_AddV2_115/AddV2_115:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_356/Mul_356?
"dense_462/Tensordot/ReadVariableOpReadVariableOp+dense_462_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_462/Tensordot/ReadVariableOp~
dense_462/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_462/Tensordot/axes?
dense_462/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_462/Tensordot/free?
dense_462/Tensordot/ShapeShapedense_461/Relu:activations:0*
T0*
_output_shapes
:2
dense_462/Tensordot/Shape?
!dense_462/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_462/Tensordot/GatherV2/axis?
dense_462/Tensordot/GatherV2GatherV2"dense_462/Tensordot/Shape:output:0!dense_462/Tensordot/free:output:0*dense_462/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_462/Tensordot/GatherV2?
#dense_462/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_462/Tensordot/GatherV2_1/axis?
dense_462/Tensordot/GatherV2_1GatherV2"dense_462/Tensordot/Shape:output:0!dense_462/Tensordot/axes:output:0,dense_462/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_462/Tensordot/GatherV2_1?
dense_462/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_462/Tensordot/Const?
dense_462/Tensordot/ProdProd%dense_462/Tensordot/GatherV2:output:0"dense_462/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_462/Tensordot/Prod?
dense_462/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_462/Tensordot/Const_1?
dense_462/Tensordot/Prod_1Prod'dense_462/Tensordot/GatherV2_1:output:0$dense_462/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_462/Tensordot/Prod_1?
dense_462/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_462/Tensordot/concat/axis?
dense_462/Tensordot/concatConcatV2!dense_462/Tensordot/free:output:0!dense_462/Tensordot/axes:output:0(dense_462/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_462/Tensordot/concat?
dense_462/Tensordot/stackPack!dense_462/Tensordot/Prod:output:0#dense_462/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_462/Tensordot/stack?
dense_462/Tensordot/transpose	Transposedense_461/Relu:activations:0#dense_462/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_462/Tensordot/transpose?
dense_462/Tensordot/ReshapeReshape!dense_462/Tensordot/transpose:y:0"dense_462/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_462/Tensordot/Reshape?
dense_462/Tensordot/MatMulMatMul$dense_462/Tensordot/Reshape:output:0*dense_462/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_462/Tensordot/MatMul?
dense_462/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_462/Tensordot/Const_2?
!dense_462/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_462/Tensordot/concat_1/axis?
dense_462/Tensordot/concat_1ConcatV2%dense_462/Tensordot/GatherV2:output:0$dense_462/Tensordot/Const_2:output:0*dense_462/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_462/Tensordot/concat_1?
dense_462/TensordotReshape$dense_462/Tensordot/MatMul:product:0%dense_462/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_462/Tensordot?
 dense_462/BiasAdd/ReadVariableOpReadVariableOp)dense_462_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_462/BiasAdd/ReadVariableOp?
dense_462/BiasAddAdddense_462/Tensordot:output:0(dense_462/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_462/BiasAddv
dense_462/ReluReludense_462/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_462/Relu?
"dense_463/Tensordot/ReadVariableOpReadVariableOp+dense_463_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_463/Tensordot/ReadVariableOp~
dense_463/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_463/Tensordot/axes?
dense_463/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_463/Tensordot/free?
dense_463/Tensordot/ShapeShapedense_462/Relu:activations:0*
T0*
_output_shapes
:2
dense_463/Tensordot/Shape?
!dense_463/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_463/Tensordot/GatherV2/axis?
dense_463/Tensordot/GatherV2GatherV2"dense_463/Tensordot/Shape:output:0!dense_463/Tensordot/free:output:0*dense_463/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_463/Tensordot/GatherV2?
#dense_463/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_463/Tensordot/GatherV2_1/axis?
dense_463/Tensordot/GatherV2_1GatherV2"dense_463/Tensordot/Shape:output:0!dense_463/Tensordot/axes:output:0,dense_463/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_463/Tensordot/GatherV2_1?
dense_463/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_463/Tensordot/Const?
dense_463/Tensordot/ProdProd%dense_463/Tensordot/GatherV2:output:0"dense_463/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_463/Tensordot/Prod?
dense_463/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_463/Tensordot/Const_1?
dense_463/Tensordot/Prod_1Prod'dense_463/Tensordot/GatherV2_1:output:0$dense_463/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_463/Tensordot/Prod_1?
dense_463/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_463/Tensordot/concat/axis?
dense_463/Tensordot/concatConcatV2!dense_463/Tensordot/free:output:0!dense_463/Tensordot/axes:output:0(dense_463/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_463/Tensordot/concat?
dense_463/Tensordot/stackPack!dense_463/Tensordot/Prod:output:0#dense_463/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_463/Tensordot/stack?
dense_463/Tensordot/transpose	Transposedense_462/Relu:activations:0#dense_463/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_463/Tensordot/transpose?
dense_463/Tensordot/ReshapeReshape!dense_463/Tensordot/transpose:y:0"dense_463/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_463/Tensordot/Reshape?
dense_463/Tensordot/MatMulMatMul$dense_463/Tensordot/Reshape:output:0*dense_463/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_463/Tensordot/MatMul?
dense_463/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_463/Tensordot/Const_2?
!dense_463/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_463/Tensordot/concat_1/axis?
dense_463/Tensordot/concat_1ConcatV2%dense_463/Tensordot/GatherV2:output:0$dense_463/Tensordot/Const_2:output:0*dense_463/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_463/Tensordot/concat_1?
dense_463/TensordotReshape$dense_463/Tensordot/MatMul:product:0%dense_463/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_463/Tensordot?
 dense_463/BiasAdd/ReadVariableOpReadVariableOp)dense_463_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_463/BiasAdd/ReadVariableOp?
dense_463/BiasAddAdddense_463/Tensordot:output:0(dense_463/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_463/BiasAdd
tf_op_layer_Pow_57/Pow_57/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
tf_op_layer_Pow_57/Pow_57/x?
tf_op_layer_Pow_57/Pow_57Pow$tf_op_layer_Pow_57/Pow_57/x:output:0tf_op_layer_Mul_356/Mul_356:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Pow_57/Pow_57?
tf_op_layer_Mul_357/Mul_357Muldense_463/BiasAdd:z:0tf_op_layer_Pow_57/Pow_57:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_357/Mul_357?
tf_op_layer_Relu_53/Relu_53Relutf_op_layer_Mul_357/Mul_357:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Relu_53/Relu_53?
+tf_op_layer_Max_61/Max_61/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+tf_op_layer_Max_61/Max_61/reduction_indices?
tf_op_layer_Max_61/Max_61Maxinputs_24tf_op_layer_Max_61/Max_61/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
tf_op_layer_Max_61/Max_61?
tf_op_layer_Mul_358/Mul_358Mul)tf_op_layer_Relu_53/Relu_53:activations:0"tf_op_layer_Max_61/Max_61:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_358/Mul_358x
IdentityIdentitytf_op_layer_Mul_358/Mul_358:z:0*
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
y
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_457068

inputs
inputs_1
identityq
Mul_357Mulinputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_357d
IdentityIdentityMul_357:z:0*
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
}
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_457747
inputs_0
inputs_1
identityx
	AddV2_114AddV2inputs_0inputs_1*
T0*
_cloned(*+
_output_shapes
:????????? 2
	AddV2_114e
IdentityIdentityAddV2_114:z:0*
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
?
?
E__inference_dense_462_layer_call_and_return_conditional_losses_457949

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
?
?
E__inference_color_law_layer_call_and_return_conditional_losses_456804

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
P
4__inference_tf_op_layer_Relu_53_layer_call_fn_458041

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
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_4570822
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
?
k
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_456947

inputs
identity[
	Mul_356/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
	Mul_356/x{
Mul_356MulMul_356/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_356d
IdentityIdentityMul_356:z:0*
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
*__inference_dense_463_layer_call_fn_458008

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
E__inference_dense_463_layer_call_and_return_conditional_losses_4570322
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
?
?
*__inference_model_115_layer_call_fn_457232
latent_params
conditional_params
	input_232
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
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_232unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
E__inference_model_115_layer_call_and_return_conditional_losses_4572112
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
_user_specified_name	input_232:
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
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_456885

inputs
inputs_1
identityw
	AddV2_115AddV2inputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2
	AddV2_115f
IdentityIdentityAddV2_115:z:0*
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
?
*__inference_model_115_layer_call_fn_457690
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
E__inference_model_115_layer_call_and_return_conditional_losses_4572112
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
?
u
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_456824

inputs
identity?
strided_slice_465/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_465/begin
strided_slice_465/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_465/end?
strided_slice_465/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_465/strides?
strided_slice_465StridedSliceinputs strided_slice_465/begin:output:0strided_slice_465/end:output:0"strided_slice_465/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_465r
IdentityIdentitystrided_slice_465:output:0*
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
L__inference_repeat_vector_57_layer_call_and_return_conditional_losses_456685

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
?
E__inference_dense_461_layer_call_and_return_conditional_losses_457897

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
?(
?
__inference__traced_save_458120
file_prefix/
+savev2_dense_460_kernel_read_readvariableop-
)savev2_dense_460_bias_read_readvariableop2
.savev2_color_law_60_kernel_read_readvariableop/
+savev2_dense_461_kernel_read_readvariableop-
)savev2_dense_461_bias_read_readvariableop/
+savev2_dense_462_kernel_read_readvariableop-
)savev2_dense_462_bias_read_readvariableop/
+savev2_dense_463_kernel_read_readvariableop-
)savev2_dense_463_bias_read_readvariableop
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
value3B1 B+_temp_10b0d02af404496d806a1ec7e0c541f1/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_460_kernel_read_readvariableop)savev2_dense_460_bias_read_readvariableop.savev2_color_law_60_kernel_read_readvariableop+savev2_dense_461_kernel_read_readvariableop)savev2_dense_461_bias_read_readvariableop+savev2_dense_462_kernel_read_readvariableop)savev2_dense_462_bias_read_readvariableop+savev2_dense_463_kernel_read_readvariableop)savev2_dense_463_bias_read_readvariableop"/device:CPU:0*
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
?
j
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_457096

inputs
identity
Max_61/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max_61/reduction_indices?
Max_61Maxinputs!Max_61/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
Max_61g
IdentityIdentityMax_61:output:0*
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
?
k
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_457082

inputs
identityh
Relu_53Reluinputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Relu_53n
IdentityIdentityRelu_53:activations:0*
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
{
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_458058
inputs_0
inputs_1
identitys
Mul_358Mulinputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_358d
IdentityIdentityMul_358:z:0*
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
j
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_457054

inputs
identityY
Pow_57/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

Pow_57/xx
Pow_57PowPow_57/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2
Pow_57c
IdentityIdentity
Pow_57:z:0*
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
*__inference_dense_461_layer_call_fn_457906

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
E__inference_dense_461_layer_call_and_return_conditional_losses_4569252
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
?
Z
>__inference_tf_op_layer_strided_slice_464_layer_call_fn_457728

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
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_4567062
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
?
}
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_457912
inputs_0
inputs_1
identityy
	AddV2_115AddV2inputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2
	AddV2_115f
IdentityIdentityAddV2_115:z:0*
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
6__inference_tf_op_layer_AddV2_114_layer_call_fn_457753
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
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_4567362
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
inputs/1
?
?
E__inference_color_law_layer_call_and_return_conditional_losses_457846

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
?K
?
E__inference_model_115_layer_call_and_return_conditional_losses_457211

inputs
inputs_1
inputs_2
color_law_457179
dense_460_457183
dense_460_457185
dense_461_457189
dense_461_457191
dense_462_457195
dense_462_457197
dense_463_457200
dense_463_457202
identity??!color_law/StatefulPartitionedCall?!dense_460/StatefulPartitionedCall?!dense_461/StatefulPartitionedCall?!dense_462/StatefulPartitionedCall?!dense_463/StatefulPartitionedCall?
 repeat_vector_57/PartitionedCallPartitionedCallinputs*
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
L__inference_repeat_vector_57_layer_call_and_return_conditional_losses_4566852"
 repeat_vector_57/PartitionedCall?
-tf_op_layer_strided_slice_464/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_4567062/
-tf_op_layer_strided_slice_464/PartitionedCall?
-tf_op_layer_strided_slice_467/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_4567222/
-tf_op_layer_strided_slice_467/PartitionedCall?
%tf_op_layer_AddV2_114/PartitionedCallPartitionedCallinputs_16tf_op_layer_strided_slice_464/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_4567362'
%tf_op_layer_AddV2_114/PartitionedCall?
-tf_op_layer_strided_slice_466/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_4567532/
-tf_op_layer_strided_slice_466/PartitionedCall?
concatenate_173/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_467/PartitionedCall:output:0.tf_op_layer_AddV2_114/PartitionedCall:output:0*
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
K__inference_concatenate_173_layer_call_and_return_conditional_losses_4567682!
concatenate_173/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_466/PartitionedCall:output:0color_law_457179*
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
E__inference_color_law_layer_call_and_return_conditional_losses_4568042#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_465/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_4568242/
-tf_op_layer_strided_slice_465/PartitionedCall?
!dense_460/StatefulPartitionedCallStatefulPartitionedCall(concatenate_173/PartitionedCall:output:0dense_460_457183dense_460_457185*
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
E__inference_dense_460_layer_call_and_return_conditional_losses_4568632#
!dense_460/StatefulPartitionedCall?
%tf_op_layer_AddV2_115/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_465/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_4568852'
%tf_op_layer_AddV2_115/PartitionedCall?
!dense_461/StatefulPartitionedCallStatefulPartitionedCall*dense_460/StatefulPartitionedCall:output:0dense_461_457189dense_461_457191*
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
E__inference_dense_461_layer_call_and_return_conditional_losses_4569252#
!dense_461/StatefulPartitionedCall?
#tf_op_layer_Mul_356/PartitionedCallPartitionedCall.tf_op_layer_AddV2_115/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4569472%
#tf_op_layer_Mul_356/PartitionedCall?
!dense_462/StatefulPartitionedCallStatefulPartitionedCall*dense_461/StatefulPartitionedCall:output:0dense_462_457195dense_462_457197*
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
E__inference_dense_462_layer_call_and_return_conditional_losses_4569862#
!dense_462/StatefulPartitionedCall?
!dense_463/StatefulPartitionedCallStatefulPartitionedCall*dense_462/StatefulPartitionedCall:output:0dense_463_457200dense_463_457202*
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
E__inference_dense_463_layer_call_and_return_conditional_losses_4570322#
!dense_463/StatefulPartitionedCall?
"tf_op_layer_Pow_57/PartitionedCallPartitionedCall,tf_op_layer_Mul_356/PartitionedCall:output:0*
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
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_4570542$
"tf_op_layer_Pow_57/PartitionedCall?
#tf_op_layer_Mul_357/PartitionedCallPartitionedCall*dense_463/StatefulPartitionedCall:output:0+tf_op_layer_Pow_57/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_4570682%
#tf_op_layer_Mul_357/PartitionedCall?
#tf_op_layer_Relu_53/PartitionedCallPartitionedCall,tf_op_layer_Mul_357/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_4570822%
#tf_op_layer_Relu_53/PartitionedCall?
"tf_op_layer_Max_61/PartitionedCallPartitionedCallinputs_2*
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
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_4570962$
"tf_op_layer_Max_61/PartitionedCall?
#tf_op_layer_Mul_358/PartitionedCallPartitionedCall,tf_op_layer_Relu_53/PartitionedCall:output:0+tf_op_layer_Max_61/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_4571102%
#tf_op_layer_Mul_358/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_358/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_460/StatefulPartitionedCall"^dense_461/StatefulPartitionedCall"^dense_462/StatefulPartitionedCall"^dense_463/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_460/StatefulPartitionedCall!dense_460/StatefulPartitionedCall2F
!dense_461/StatefulPartitionedCall!dense_461/StatefulPartitionedCall2F
!dense_462/StatefulPartitionedCall!dense_462/StatefulPartitionedCall2F
!dense_463/StatefulPartitionedCall!dense_463/StatefulPartitionedCall:O K
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
??
?
!__inference__wrapped_model_456676
latent_params
conditional_params
	input_2329
5model_115_color_law_tensordot_readvariableop_resource9
5model_115_dense_460_tensordot_readvariableop_resource7
3model_115_dense_460_biasadd_readvariableop_resource9
5model_115_dense_461_tensordot_readvariableop_resource7
3model_115_dense_461_biasadd_readvariableop_resource9
5model_115_dense_462_tensordot_readvariableop_resource7
3model_115_dense_462_biasadd_readvariableop_resource9
5model_115_dense_463_tensordot_readvariableop_resource7
3model_115_dense_463_biasadd_readvariableop_resource
identity??
)model_115/repeat_vector_57/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_115/repeat_vector_57/ExpandDims/dim?
%model_115/repeat_vector_57/ExpandDims
ExpandDimslatent_params2model_115/repeat_vector_57/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_115/repeat_vector_57/ExpandDims?
 model_115/repeat_vector_57/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_115/repeat_vector_57/stack?
model_115/repeat_vector_57/TileTile.model_115/repeat_vector_57/ExpandDims:output:0)model_115/repeat_vector_57/stack:output:0*
T0*+
_output_shapes
:????????? 2!
model_115/repeat_vector_57/Tile?
?model_115/tf_op_layer_strided_slice_464/strided_slice_464/beginConst*
_output_shapes
:*
dtype0*
valueB"        2A
?model_115/tf_op_layer_strided_slice_464/strided_slice_464/begin?
=model_115/tf_op_layer_strided_slice_464/strided_slice_464/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_115/tf_op_layer_strided_slice_464/strided_slice_464/end?
Amodel_115/tf_op_layer_strided_slice_464/strided_slice_464/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_115/tf_op_layer_strided_slice_464/strided_slice_464/strides?
9model_115/tf_op_layer_strided_slice_464/strided_slice_464StridedSlice(model_115/repeat_vector_57/Tile:output:0Hmodel_115/tf_op_layer_strided_slice_464/strided_slice_464/begin:output:0Fmodel_115/tf_op_layer_strided_slice_464/strided_slice_464/end:output:0Jmodel_115/tf_op_layer_strided_slice_464/strided_slice_464/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_115/tf_op_layer_strided_slice_464/strided_slice_464?
?model_115/tf_op_layer_strided_slice_467/strided_slice_467/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_115/tf_op_layer_strided_slice_467/strided_slice_467/begin?
=model_115/tf_op_layer_strided_slice_467/strided_slice_467/endConst*
_output_shapes
:*
dtype0*
valueB"        2?
=model_115/tf_op_layer_strided_slice_467/strided_slice_467/end?
Amodel_115/tf_op_layer_strided_slice_467/strided_slice_467/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_115/tf_op_layer_strided_slice_467/strided_slice_467/strides?
9model_115/tf_op_layer_strided_slice_467/strided_slice_467StridedSlice(model_115/repeat_vector_57/Tile:output:0Hmodel_115/tf_op_layer_strided_slice_467/strided_slice_467/begin:output:0Fmodel_115/tf_op_layer_strided_slice_467/strided_slice_467/end:output:0Jmodel_115/tf_op_layer_strided_slice_467/strided_slice_467/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2;
9model_115/tf_op_layer_strided_slice_467/strided_slice_467?
)model_115/tf_op_layer_AddV2_114/AddV2_114AddV2conditional_paramsBmodel_115/tf_op_layer_strided_slice_464/strided_slice_464:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2+
)model_115/tf_op_layer_AddV2_114/AddV2_114?
?model_115/tf_op_layer_strided_slice_466/strided_slice_466/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_115/tf_op_layer_strided_slice_466/strided_slice_466/begin?
=model_115/tf_op_layer_strided_slice_466/strided_slice_466/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_115/tf_op_layer_strided_slice_466/strided_slice_466/end?
Amodel_115/tf_op_layer_strided_slice_466/strided_slice_466/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_115/tf_op_layer_strided_slice_466/strided_slice_466/strides?
9model_115/tf_op_layer_strided_slice_466/strided_slice_466StridedSlice(model_115/repeat_vector_57/Tile:output:0Hmodel_115/tf_op_layer_strided_slice_466/strided_slice_466/begin:output:0Fmodel_115/tf_op_layer_strided_slice_466/strided_slice_466/end:output:0Jmodel_115/tf_op_layer_strided_slice_466/strided_slice_466/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_115/tf_op_layer_strided_slice_466/strided_slice_466?
%model_115/concatenate_173/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_115/concatenate_173/concat/axis?
 model_115/concatenate_173/concatConcatV2Bmodel_115/tf_op_layer_strided_slice_467/strided_slice_467:output:0-model_115/tf_op_layer_AddV2_114/AddV2_114:z:0.model_115/concatenate_173/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2"
 model_115/concatenate_173/concat?
,model_115/color_law/Tensordot/ReadVariableOpReadVariableOp5model_115_color_law_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,model_115/color_law/Tensordot/ReadVariableOp?
"model_115/color_law/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_115/color_law/Tensordot/axes?
"model_115/color_law/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_115/color_law/Tensordot/free?
#model_115/color_law/Tensordot/ShapeShapeBmodel_115/tf_op_layer_strided_slice_466/strided_slice_466:output:0*
T0*
_output_shapes
:2%
#model_115/color_law/Tensordot/Shape?
+model_115/color_law/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/color_law/Tensordot/GatherV2/axis?
&model_115/color_law/Tensordot/GatherV2GatherV2,model_115/color_law/Tensordot/Shape:output:0+model_115/color_law/Tensordot/free:output:04model_115/color_law/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_115/color_law/Tensordot/GatherV2?
-model_115/color_law/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_115/color_law/Tensordot/GatherV2_1/axis?
(model_115/color_law/Tensordot/GatherV2_1GatherV2,model_115/color_law/Tensordot/Shape:output:0+model_115/color_law/Tensordot/axes:output:06model_115/color_law/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_115/color_law/Tensordot/GatherV2_1?
#model_115/color_law/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_115/color_law/Tensordot/Const?
"model_115/color_law/Tensordot/ProdProd/model_115/color_law/Tensordot/GatherV2:output:0,model_115/color_law/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_115/color_law/Tensordot/Prod?
%model_115/color_law/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_115/color_law/Tensordot/Const_1?
$model_115/color_law/Tensordot/Prod_1Prod1model_115/color_law/Tensordot/GatherV2_1:output:0.model_115/color_law/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_115/color_law/Tensordot/Prod_1?
)model_115/color_law/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_115/color_law/Tensordot/concat/axis?
$model_115/color_law/Tensordot/concatConcatV2+model_115/color_law/Tensordot/free:output:0+model_115/color_law/Tensordot/axes:output:02model_115/color_law/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_115/color_law/Tensordot/concat?
#model_115/color_law/Tensordot/stackPack+model_115/color_law/Tensordot/Prod:output:0-model_115/color_law/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_115/color_law/Tensordot/stack?
'model_115/color_law/Tensordot/transpose	TransposeBmodel_115/tf_op_layer_strided_slice_466/strided_slice_466:output:0-model_115/color_law/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2)
'model_115/color_law/Tensordot/transpose?
%model_115/color_law/Tensordot/ReshapeReshape+model_115/color_law/Tensordot/transpose:y:0,model_115/color_law/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_115/color_law/Tensordot/Reshape?
$model_115/color_law/Tensordot/MatMulMatMul.model_115/color_law/Tensordot/Reshape:output:04model_115/color_law/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_115/color_law/Tensordot/MatMul?
%model_115/color_law/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_115/color_law/Tensordot/Const_2?
+model_115/color_law/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/color_law/Tensordot/concat_1/axis?
&model_115/color_law/Tensordot/concat_1ConcatV2/model_115/color_law/Tensordot/GatherV2:output:0.model_115/color_law/Tensordot/Const_2:output:04model_115/color_law/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_115/color_law/Tensordot/concat_1?
model_115/color_law/TensordotReshape.model_115/color_law/Tensordot/MatMul:product:0/model_115/color_law/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_115/color_law/Tensordot?
?model_115/tf_op_layer_strided_slice_465/strided_slice_465/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_115/tf_op_layer_strided_slice_465/strided_slice_465/begin?
=model_115/tf_op_layer_strided_slice_465/strided_slice_465/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_115/tf_op_layer_strided_slice_465/strided_slice_465/end?
Amodel_115/tf_op_layer_strided_slice_465/strided_slice_465/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_115/tf_op_layer_strided_slice_465/strided_slice_465/strides?
9model_115/tf_op_layer_strided_slice_465/strided_slice_465StridedSlice(model_115/repeat_vector_57/Tile:output:0Hmodel_115/tf_op_layer_strided_slice_465/strided_slice_465/begin:output:0Fmodel_115/tf_op_layer_strided_slice_465/strided_slice_465/end:output:0Jmodel_115/tf_op_layer_strided_slice_465/strided_slice_465/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_115/tf_op_layer_strided_slice_465/strided_slice_465?
,model_115/dense_460/Tensordot/ReadVariableOpReadVariableOp5model_115_dense_460_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02.
,model_115/dense_460/Tensordot/ReadVariableOp?
"model_115/dense_460/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_115/dense_460/Tensordot/axes?
"model_115/dense_460/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_115/dense_460/Tensordot/free?
#model_115/dense_460/Tensordot/ShapeShape)model_115/concatenate_173/concat:output:0*
T0*
_output_shapes
:2%
#model_115/dense_460/Tensordot/Shape?
+model_115/dense_460/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/dense_460/Tensordot/GatherV2/axis?
&model_115/dense_460/Tensordot/GatherV2GatherV2,model_115/dense_460/Tensordot/Shape:output:0+model_115/dense_460/Tensordot/free:output:04model_115/dense_460/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_115/dense_460/Tensordot/GatherV2?
-model_115/dense_460/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_115/dense_460/Tensordot/GatherV2_1/axis?
(model_115/dense_460/Tensordot/GatherV2_1GatherV2,model_115/dense_460/Tensordot/Shape:output:0+model_115/dense_460/Tensordot/axes:output:06model_115/dense_460/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_115/dense_460/Tensordot/GatherV2_1?
#model_115/dense_460/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_115/dense_460/Tensordot/Const?
"model_115/dense_460/Tensordot/ProdProd/model_115/dense_460/Tensordot/GatherV2:output:0,model_115/dense_460/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_115/dense_460/Tensordot/Prod?
%model_115/dense_460/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_115/dense_460/Tensordot/Const_1?
$model_115/dense_460/Tensordot/Prod_1Prod1model_115/dense_460/Tensordot/GatherV2_1:output:0.model_115/dense_460/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_115/dense_460/Tensordot/Prod_1?
)model_115/dense_460/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_115/dense_460/Tensordot/concat/axis?
$model_115/dense_460/Tensordot/concatConcatV2+model_115/dense_460/Tensordot/free:output:0+model_115/dense_460/Tensordot/axes:output:02model_115/dense_460/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_115/dense_460/Tensordot/concat?
#model_115/dense_460/Tensordot/stackPack+model_115/dense_460/Tensordot/Prod:output:0-model_115/dense_460/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_115/dense_460/Tensordot/stack?
'model_115/dense_460/Tensordot/transpose	Transpose)model_115/concatenate_173/concat:output:0-model_115/dense_460/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2)
'model_115/dense_460/Tensordot/transpose?
%model_115/dense_460/Tensordot/ReshapeReshape+model_115/dense_460/Tensordot/transpose:y:0,model_115/dense_460/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_115/dense_460/Tensordot/Reshape?
$model_115/dense_460/Tensordot/MatMulMatMul.model_115/dense_460/Tensordot/Reshape:output:04model_115/dense_460/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$model_115/dense_460/Tensordot/MatMul?
%model_115/dense_460/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_115/dense_460/Tensordot/Const_2?
+model_115/dense_460/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/dense_460/Tensordot/concat_1/axis?
&model_115/dense_460/Tensordot/concat_1ConcatV2/model_115/dense_460/Tensordot/GatherV2:output:0.model_115/dense_460/Tensordot/Const_2:output:04model_115/dense_460/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_115/dense_460/Tensordot/concat_1?
model_115/dense_460/TensordotReshape.model_115/dense_460/Tensordot/MatMul:product:0/model_115/dense_460/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
model_115/dense_460/Tensordot?
*model_115/dense_460/BiasAdd/ReadVariableOpReadVariableOp3model_115_dense_460_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*model_115/dense_460/BiasAdd/ReadVariableOp?
model_115/dense_460/BiasAddAdd&model_115/dense_460/Tensordot:output:02model_115/dense_460/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
model_115/dense_460/BiasAdd?
model_115/dense_460/ReluRelumodel_115/dense_460/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
model_115/dense_460/Relu?
)model_115/tf_op_layer_AddV2_115/AddV2_115AddV2&model_115/color_law/Tensordot:output:0Bmodel_115/tf_op_layer_strided_slice_465/strided_slice_465:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2+
)model_115/tf_op_layer_AddV2_115/AddV2_115?
,model_115/dense_461/Tensordot/ReadVariableOpReadVariableOp5model_115_dense_461_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02.
,model_115/dense_461/Tensordot/ReadVariableOp?
"model_115/dense_461/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_115/dense_461/Tensordot/axes?
"model_115/dense_461/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_115/dense_461/Tensordot/free?
#model_115/dense_461/Tensordot/ShapeShape&model_115/dense_460/Relu:activations:0*
T0*
_output_shapes
:2%
#model_115/dense_461/Tensordot/Shape?
+model_115/dense_461/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/dense_461/Tensordot/GatherV2/axis?
&model_115/dense_461/Tensordot/GatherV2GatherV2,model_115/dense_461/Tensordot/Shape:output:0+model_115/dense_461/Tensordot/free:output:04model_115/dense_461/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_115/dense_461/Tensordot/GatherV2?
-model_115/dense_461/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_115/dense_461/Tensordot/GatherV2_1/axis?
(model_115/dense_461/Tensordot/GatherV2_1GatherV2,model_115/dense_461/Tensordot/Shape:output:0+model_115/dense_461/Tensordot/axes:output:06model_115/dense_461/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_115/dense_461/Tensordot/GatherV2_1?
#model_115/dense_461/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_115/dense_461/Tensordot/Const?
"model_115/dense_461/Tensordot/ProdProd/model_115/dense_461/Tensordot/GatherV2:output:0,model_115/dense_461/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_115/dense_461/Tensordot/Prod?
%model_115/dense_461/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_115/dense_461/Tensordot/Const_1?
$model_115/dense_461/Tensordot/Prod_1Prod1model_115/dense_461/Tensordot/GatherV2_1:output:0.model_115/dense_461/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_115/dense_461/Tensordot/Prod_1?
)model_115/dense_461/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_115/dense_461/Tensordot/concat/axis?
$model_115/dense_461/Tensordot/concatConcatV2+model_115/dense_461/Tensordot/free:output:0+model_115/dense_461/Tensordot/axes:output:02model_115/dense_461/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_115/dense_461/Tensordot/concat?
#model_115/dense_461/Tensordot/stackPack+model_115/dense_461/Tensordot/Prod:output:0-model_115/dense_461/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_115/dense_461/Tensordot/stack?
'model_115/dense_461/Tensordot/transpose	Transpose&model_115/dense_460/Relu:activations:0-model_115/dense_461/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2)
'model_115/dense_461/Tensordot/transpose?
%model_115/dense_461/Tensordot/ReshapeReshape+model_115/dense_461/Tensordot/transpose:y:0,model_115/dense_461/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_115/dense_461/Tensordot/Reshape?
$model_115/dense_461/Tensordot/MatMulMatMul.model_115/dense_461/Tensordot/Reshape:output:04model_115/dense_461/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_115/dense_461/Tensordot/MatMul?
%model_115/dense_461/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_115/dense_461/Tensordot/Const_2?
+model_115/dense_461/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/dense_461/Tensordot/concat_1/axis?
&model_115/dense_461/Tensordot/concat_1ConcatV2/model_115/dense_461/Tensordot/GatherV2:output:0.model_115/dense_461/Tensordot/Const_2:output:04model_115/dense_461/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_115/dense_461/Tensordot/concat_1?
model_115/dense_461/TensordotReshape.model_115/dense_461/Tensordot/MatMul:product:0/model_115/dense_461/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_115/dense_461/Tensordot?
*model_115/dense_461/BiasAdd/ReadVariableOpReadVariableOp3model_115_dense_461_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_115/dense_461/BiasAdd/ReadVariableOp?
model_115/dense_461/BiasAddAdd&model_115/dense_461/Tensordot:output:02model_115/dense_461/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_115/dense_461/BiasAdd?
model_115/dense_461/ReluRelumodel_115/dense_461/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
model_115/dense_461/Relu?
'model_115/tf_op_layer_Mul_356/Mul_356/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2)
'model_115/tf_op_layer_Mul_356/Mul_356/x?
%model_115/tf_op_layer_Mul_356/Mul_356Mul0model_115/tf_op_layer_Mul_356/Mul_356/x:output:0-model_115/tf_op_layer_AddV2_115/AddV2_115:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_115/tf_op_layer_Mul_356/Mul_356?
,model_115/dense_462/Tensordot/ReadVariableOpReadVariableOp5model_115_dense_462_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,model_115/dense_462/Tensordot/ReadVariableOp?
"model_115/dense_462/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_115/dense_462/Tensordot/axes?
"model_115/dense_462/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_115/dense_462/Tensordot/free?
#model_115/dense_462/Tensordot/ShapeShape&model_115/dense_461/Relu:activations:0*
T0*
_output_shapes
:2%
#model_115/dense_462/Tensordot/Shape?
+model_115/dense_462/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/dense_462/Tensordot/GatherV2/axis?
&model_115/dense_462/Tensordot/GatherV2GatherV2,model_115/dense_462/Tensordot/Shape:output:0+model_115/dense_462/Tensordot/free:output:04model_115/dense_462/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_115/dense_462/Tensordot/GatherV2?
-model_115/dense_462/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_115/dense_462/Tensordot/GatherV2_1/axis?
(model_115/dense_462/Tensordot/GatherV2_1GatherV2,model_115/dense_462/Tensordot/Shape:output:0+model_115/dense_462/Tensordot/axes:output:06model_115/dense_462/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_115/dense_462/Tensordot/GatherV2_1?
#model_115/dense_462/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_115/dense_462/Tensordot/Const?
"model_115/dense_462/Tensordot/ProdProd/model_115/dense_462/Tensordot/GatherV2:output:0,model_115/dense_462/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_115/dense_462/Tensordot/Prod?
%model_115/dense_462/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_115/dense_462/Tensordot/Const_1?
$model_115/dense_462/Tensordot/Prod_1Prod1model_115/dense_462/Tensordot/GatherV2_1:output:0.model_115/dense_462/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_115/dense_462/Tensordot/Prod_1?
)model_115/dense_462/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_115/dense_462/Tensordot/concat/axis?
$model_115/dense_462/Tensordot/concatConcatV2+model_115/dense_462/Tensordot/free:output:0+model_115/dense_462/Tensordot/axes:output:02model_115/dense_462/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_115/dense_462/Tensordot/concat?
#model_115/dense_462/Tensordot/stackPack+model_115/dense_462/Tensordot/Prod:output:0-model_115/dense_462/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_115/dense_462/Tensordot/stack?
'model_115/dense_462/Tensordot/transpose	Transpose&model_115/dense_461/Relu:activations:0-model_115/dense_462/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2)
'model_115/dense_462/Tensordot/transpose?
%model_115/dense_462/Tensordot/ReshapeReshape+model_115/dense_462/Tensordot/transpose:y:0,model_115/dense_462/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_115/dense_462/Tensordot/Reshape?
$model_115/dense_462/Tensordot/MatMulMatMul.model_115/dense_462/Tensordot/Reshape:output:04model_115/dense_462/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_115/dense_462/Tensordot/MatMul?
%model_115/dense_462/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_115/dense_462/Tensordot/Const_2?
+model_115/dense_462/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/dense_462/Tensordot/concat_1/axis?
&model_115/dense_462/Tensordot/concat_1ConcatV2/model_115/dense_462/Tensordot/GatherV2:output:0.model_115/dense_462/Tensordot/Const_2:output:04model_115/dense_462/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_115/dense_462/Tensordot/concat_1?
model_115/dense_462/TensordotReshape.model_115/dense_462/Tensordot/MatMul:product:0/model_115/dense_462/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_115/dense_462/Tensordot?
*model_115/dense_462/BiasAdd/ReadVariableOpReadVariableOp3model_115_dense_462_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_115/dense_462/BiasAdd/ReadVariableOp?
model_115/dense_462/BiasAddAdd&model_115/dense_462/Tensordot:output:02model_115/dense_462/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_115/dense_462/BiasAdd?
model_115/dense_462/ReluRelumodel_115/dense_462/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
model_115/dense_462/Relu?
,model_115/dense_463/Tensordot/ReadVariableOpReadVariableOp5model_115_dense_463_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,model_115/dense_463/Tensordot/ReadVariableOp?
"model_115/dense_463/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_115/dense_463/Tensordot/axes?
"model_115/dense_463/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_115/dense_463/Tensordot/free?
#model_115/dense_463/Tensordot/ShapeShape&model_115/dense_462/Relu:activations:0*
T0*
_output_shapes
:2%
#model_115/dense_463/Tensordot/Shape?
+model_115/dense_463/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/dense_463/Tensordot/GatherV2/axis?
&model_115/dense_463/Tensordot/GatherV2GatherV2,model_115/dense_463/Tensordot/Shape:output:0+model_115/dense_463/Tensordot/free:output:04model_115/dense_463/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_115/dense_463/Tensordot/GatherV2?
-model_115/dense_463/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_115/dense_463/Tensordot/GatherV2_1/axis?
(model_115/dense_463/Tensordot/GatherV2_1GatherV2,model_115/dense_463/Tensordot/Shape:output:0+model_115/dense_463/Tensordot/axes:output:06model_115/dense_463/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_115/dense_463/Tensordot/GatherV2_1?
#model_115/dense_463/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_115/dense_463/Tensordot/Const?
"model_115/dense_463/Tensordot/ProdProd/model_115/dense_463/Tensordot/GatherV2:output:0,model_115/dense_463/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_115/dense_463/Tensordot/Prod?
%model_115/dense_463/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_115/dense_463/Tensordot/Const_1?
$model_115/dense_463/Tensordot/Prod_1Prod1model_115/dense_463/Tensordot/GatherV2_1:output:0.model_115/dense_463/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_115/dense_463/Tensordot/Prod_1?
)model_115/dense_463/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_115/dense_463/Tensordot/concat/axis?
$model_115/dense_463/Tensordot/concatConcatV2+model_115/dense_463/Tensordot/free:output:0+model_115/dense_463/Tensordot/axes:output:02model_115/dense_463/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_115/dense_463/Tensordot/concat?
#model_115/dense_463/Tensordot/stackPack+model_115/dense_463/Tensordot/Prod:output:0-model_115/dense_463/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_115/dense_463/Tensordot/stack?
'model_115/dense_463/Tensordot/transpose	Transpose&model_115/dense_462/Relu:activations:0-model_115/dense_463/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2)
'model_115/dense_463/Tensordot/transpose?
%model_115/dense_463/Tensordot/ReshapeReshape+model_115/dense_463/Tensordot/transpose:y:0,model_115/dense_463/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_115/dense_463/Tensordot/Reshape?
$model_115/dense_463/Tensordot/MatMulMatMul.model_115/dense_463/Tensordot/Reshape:output:04model_115/dense_463/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_115/dense_463/Tensordot/MatMul?
%model_115/dense_463/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_115/dense_463/Tensordot/Const_2?
+model_115/dense_463/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_115/dense_463/Tensordot/concat_1/axis?
&model_115/dense_463/Tensordot/concat_1ConcatV2/model_115/dense_463/Tensordot/GatherV2:output:0.model_115/dense_463/Tensordot/Const_2:output:04model_115/dense_463/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_115/dense_463/Tensordot/concat_1?
model_115/dense_463/TensordotReshape.model_115/dense_463/Tensordot/MatMul:product:0/model_115/dense_463/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_115/dense_463/Tensordot?
*model_115/dense_463/BiasAdd/ReadVariableOpReadVariableOp3model_115_dense_463_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_115/dense_463/BiasAdd/ReadVariableOp?
model_115/dense_463/BiasAddAdd&model_115/dense_463/Tensordot:output:02model_115/dense_463/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_115/dense_463/BiasAdd?
%model_115/tf_op_layer_Pow_57/Pow_57/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2'
%model_115/tf_op_layer_Pow_57/Pow_57/x?
#model_115/tf_op_layer_Pow_57/Pow_57Pow.model_115/tf_op_layer_Pow_57/Pow_57/x:output:0)model_115/tf_op_layer_Mul_356/Mul_356:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2%
#model_115/tf_op_layer_Pow_57/Pow_57?
%model_115/tf_op_layer_Mul_357/Mul_357Mulmodel_115/dense_463/BiasAdd:z:0'model_115/tf_op_layer_Pow_57/Pow_57:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_115/tf_op_layer_Mul_357/Mul_357?
%model_115/tf_op_layer_Relu_53/Relu_53Relu)model_115/tf_op_layer_Mul_357/Mul_357:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_115/tf_op_layer_Relu_53/Relu_53?
5model_115/tf_op_layer_Max_61/Max_61/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5model_115/tf_op_layer_Max_61/Max_61/reduction_indices?
#model_115/tf_op_layer_Max_61/Max_61Max	input_232>model_115/tf_op_layer_Max_61/Max_61/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2%
#model_115/tf_op_layer_Max_61/Max_61?
%model_115/tf_op_layer_Mul_358/Mul_358Mul3model_115/tf_op_layer_Relu_53/Relu_53:activations:0,model_115/tf_op_layer_Max_61/Max_61:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_115/tf_op_layer_Mul_358/Mul_358?
IdentityIdentity)model_115/tf_op_layer_Mul_358/Mul_358:z:0*
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
_user_specified_name	input_232:
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
*__inference_model_115_layer_call_fn_457300
latent_params
conditional_params
	input_232
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
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_232unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
E__inference_model_115_layer_call_and_return_conditional_losses_4572792
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
_user_specified_name	input_232:
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
E__inference_model_115_layer_call_and_return_conditional_losses_457120
latent_params
conditional_params
	input_232
color_law_456813
dense_460_456874
dense_460_456876
dense_461_456936
dense_461_456938
dense_462_456997
dense_462_456999
dense_463_457043
dense_463_457045
identity??!color_law/StatefulPartitionedCall?!dense_460/StatefulPartitionedCall?!dense_461/StatefulPartitionedCall?!dense_462/StatefulPartitionedCall?!dense_463/StatefulPartitionedCall?
 repeat_vector_57/PartitionedCallPartitionedCalllatent_params*
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
L__inference_repeat_vector_57_layer_call_and_return_conditional_losses_4566852"
 repeat_vector_57/PartitionedCall?
-tf_op_layer_strided_slice_464/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_4567062/
-tf_op_layer_strided_slice_464/PartitionedCall?
-tf_op_layer_strided_slice_467/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_4567222/
-tf_op_layer_strided_slice_467/PartitionedCall?
%tf_op_layer_AddV2_114/PartitionedCallPartitionedCallconditional_params6tf_op_layer_strided_slice_464/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_4567362'
%tf_op_layer_AddV2_114/PartitionedCall?
-tf_op_layer_strided_slice_466/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_4567532/
-tf_op_layer_strided_slice_466/PartitionedCall?
concatenate_173/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_467/PartitionedCall:output:0.tf_op_layer_AddV2_114/PartitionedCall:output:0*
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
K__inference_concatenate_173_layer_call_and_return_conditional_losses_4567682!
concatenate_173/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_466/PartitionedCall:output:0color_law_456813*
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
E__inference_color_law_layer_call_and_return_conditional_losses_4568042#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_465/PartitionedCallPartitionedCall)repeat_vector_57/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_4568242/
-tf_op_layer_strided_slice_465/PartitionedCall?
!dense_460/StatefulPartitionedCallStatefulPartitionedCall(concatenate_173/PartitionedCall:output:0dense_460_456874dense_460_456876*
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
E__inference_dense_460_layer_call_and_return_conditional_losses_4568632#
!dense_460/StatefulPartitionedCall?
%tf_op_layer_AddV2_115/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_465/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_4568852'
%tf_op_layer_AddV2_115/PartitionedCall?
!dense_461/StatefulPartitionedCallStatefulPartitionedCall*dense_460/StatefulPartitionedCall:output:0dense_461_456936dense_461_456938*
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
E__inference_dense_461_layer_call_and_return_conditional_losses_4569252#
!dense_461/StatefulPartitionedCall?
#tf_op_layer_Mul_356/PartitionedCallPartitionedCall.tf_op_layer_AddV2_115/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4569472%
#tf_op_layer_Mul_356/PartitionedCall?
!dense_462/StatefulPartitionedCallStatefulPartitionedCall*dense_461/StatefulPartitionedCall:output:0dense_462_456997dense_462_456999*
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
E__inference_dense_462_layer_call_and_return_conditional_losses_4569862#
!dense_462/StatefulPartitionedCall?
!dense_463/StatefulPartitionedCallStatefulPartitionedCall*dense_462/StatefulPartitionedCall:output:0dense_463_457043dense_463_457045*
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
E__inference_dense_463_layer_call_and_return_conditional_losses_4570322#
!dense_463/StatefulPartitionedCall?
"tf_op_layer_Pow_57/PartitionedCallPartitionedCall,tf_op_layer_Mul_356/PartitionedCall:output:0*
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
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_4570542$
"tf_op_layer_Pow_57/PartitionedCall?
#tf_op_layer_Mul_357/PartitionedCallPartitionedCall*dense_463/StatefulPartitionedCall:output:0+tf_op_layer_Pow_57/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_4570682%
#tf_op_layer_Mul_357/PartitionedCall?
#tf_op_layer_Relu_53/PartitionedCallPartitionedCall,tf_op_layer_Mul_357/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_4570822%
#tf_op_layer_Relu_53/PartitionedCall?
"tf_op_layer_Max_61/PartitionedCallPartitionedCall	input_232*
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
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_4570962$
"tf_op_layer_Max_61/PartitionedCall?
#tf_op_layer_Mul_358/PartitionedCallPartitionedCall,tf_op_layer_Relu_53/PartitionedCall:output:0+tf_op_layer_Max_61/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_4571102%
#tf_op_layer_Mul_358/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_358/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_460/StatefulPartitionedCall"^dense_461/StatefulPartitionedCall"^dense_462/StatefulPartitionedCall"^dense_463/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_460/StatefulPartitionedCall!dense_460/StatefulPartitionedCall2F
!dense_461/StatefulPartitionedCall!dense_461/StatefulPartitionedCall2F
!dense_462/StatefulPartitionedCall!dense_462/StatefulPartitionedCall2F
!dense_463/StatefulPartitionedCall!dense_463/StatefulPartitionedCall:V R
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
_user_specified_name	input_232:
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
E__inference_dense_461_layer_call_and_return_conditional_losses_456925

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
?
\
0__inference_concatenate_173_layer_call_fn_457766
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
K__inference_concatenate_173_layer_call_and_return_conditional_losses_4567682
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
?
u
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_457723

inputs
identity?
strided_slice_464/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_464/begin
strided_slice_464/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_464/end?
strided_slice_464/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_464/strides?
strided_slice_464StridedSliceinputs strided_slice_464/begin:output:0strided_slice_464/end:output:0"strided_slice_464/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_464r
IdentityIdentitystrided_slice_464:output:0*
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
?
u
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_456706

inputs
identity?
strided_slice_464/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_464/begin
strided_slice_464/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_464/end?
strided_slice_464/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_464/strides?
strided_slice_464StridedSliceinputs strided_slice_464/begin:output:0strided_slice_464/end:output:0"strided_slice_464/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_464r
IdentityIdentitystrided_slice_464:output:0*
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
?
u
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_457774

inputs
identity?
strided_slice_466/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_466/begin
strided_slice_466/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_466/end?
strided_slice_466/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_466/strides?
strided_slice_466StridedSliceinputs strided_slice_466/begin:output:0strided_slice_466/end:output:0"strided_slice_466/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_466r
IdentityIdentitystrided_slice_466:output:0*
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
?
u
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_457736

inputs
identity?
strided_slice_467/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_467/begin
strided_slice_467/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_467/end?
strided_slice_467/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_467/strides?
strided_slice_467StridedSliceinputs strided_slice_467/begin:output:0strided_slice_467/end:output:0"strided_slice_467/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2
strided_slice_467r
IdentityIdentitystrided_slice_467:output:0*
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
?
`
4__inference_tf_op_layer_Mul_357_layer_call_fn_458031
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
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_4570682
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
?
j
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_458047

inputs
identity
Max_61/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max_61/reduction_indices?
Max_61Maxinputs!Max_61/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
Max_61g
IdentityIdentityMax_61:output:0*
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
??
?
E__inference_model_115_layer_call_and_return_conditional_losses_457665
inputs_0
inputs_1
inputs_2/
+color_law_tensordot_readvariableop_resource/
+dense_460_tensordot_readvariableop_resource-
)dense_460_biasadd_readvariableop_resource/
+dense_461_tensordot_readvariableop_resource-
)dense_461_biasadd_readvariableop_resource/
+dense_462_tensordot_readvariableop_resource-
)dense_462_biasadd_readvariableop_resource/
+dense_463_tensordot_readvariableop_resource-
)dense_463_biasadd_readvariableop_resource
identity??
repeat_vector_57/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_57/ExpandDims/dim?
repeat_vector_57/ExpandDims
ExpandDimsinputs_0(repeat_vector_57/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
repeat_vector_57/ExpandDims?
repeat_vector_57/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
repeat_vector_57/stack?
repeat_vector_57/TileTile$repeat_vector_57/ExpandDims:output:0repeat_vector_57/stack:output:0*
T0*+
_output_shapes
:????????? 2
repeat_vector_57/Tile?
5tf_op_layer_strided_slice_464/strided_slice_464/beginConst*
_output_shapes
:*
dtype0*
valueB"        27
5tf_op_layer_strided_slice_464/strided_slice_464/begin?
3tf_op_layer_strided_slice_464/strided_slice_464/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_464/strided_slice_464/end?
7tf_op_layer_strided_slice_464/strided_slice_464/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_464/strided_slice_464/strides?
/tf_op_layer_strided_slice_464/strided_slice_464StridedSlicerepeat_vector_57/Tile:output:0>tf_op_layer_strided_slice_464/strided_slice_464/begin:output:0<tf_op_layer_strided_slice_464/strided_slice_464/end:output:0@tf_op_layer_strided_slice_464/strided_slice_464/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_464/strided_slice_464?
5tf_op_layer_strided_slice_467/strided_slice_467/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_467/strided_slice_467/begin?
3tf_op_layer_strided_slice_467/strided_slice_467/endConst*
_output_shapes
:*
dtype0*
valueB"        25
3tf_op_layer_strided_slice_467/strided_slice_467/end?
7tf_op_layer_strided_slice_467/strided_slice_467/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_467/strided_slice_467/strides?
/tf_op_layer_strided_slice_467/strided_slice_467StridedSlicerepeat_vector_57/Tile:output:0>tf_op_layer_strided_slice_467/strided_slice_467/begin:output:0<tf_op_layer_strided_slice_467/strided_slice_467/end:output:0@tf_op_layer_strided_slice_467/strided_slice_467/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask21
/tf_op_layer_strided_slice_467/strided_slice_467?
tf_op_layer_AddV2_114/AddV2_114AddV2inputs_18tf_op_layer_strided_slice_464/strided_slice_464:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2!
tf_op_layer_AddV2_114/AddV2_114?
5tf_op_layer_strided_slice_466/strided_slice_466/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_466/strided_slice_466/begin?
3tf_op_layer_strided_slice_466/strided_slice_466/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_466/strided_slice_466/end?
7tf_op_layer_strided_slice_466/strided_slice_466/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_466/strided_slice_466/strides?
/tf_op_layer_strided_slice_466/strided_slice_466StridedSlicerepeat_vector_57/Tile:output:0>tf_op_layer_strided_slice_466/strided_slice_466/begin:output:0<tf_op_layer_strided_slice_466/strided_slice_466/end:output:0@tf_op_layer_strided_slice_466/strided_slice_466/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_466/strided_slice_466|
concatenate_173/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_173/concat/axis?
concatenate_173/concatConcatV28tf_op_layer_strided_slice_467/strided_slice_467:output:0#tf_op_layer_AddV2_114/AddV2_114:z:0$concatenate_173/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concatenate_173/concat?
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
color_law/Tensordot/ShapeShape8tf_op_layer_strided_slice_466/strided_slice_466:output:0*
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
color_law/Tensordot/transpose	Transpose8tf_op_layer_strided_slice_466/strided_slice_466:output:0#color_law/Tensordot/concat:output:0*
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
5tf_op_layer_strided_slice_465/strided_slice_465/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_465/strided_slice_465/begin?
3tf_op_layer_strided_slice_465/strided_slice_465/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_465/strided_slice_465/end?
7tf_op_layer_strided_slice_465/strided_slice_465/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_465/strided_slice_465/strides?
/tf_op_layer_strided_slice_465/strided_slice_465StridedSlicerepeat_vector_57/Tile:output:0>tf_op_layer_strided_slice_465/strided_slice_465/begin:output:0<tf_op_layer_strided_slice_465/strided_slice_465/end:output:0@tf_op_layer_strided_slice_465/strided_slice_465/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_465/strided_slice_465?
"dense_460/Tensordot/ReadVariableOpReadVariableOp+dense_460_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02$
"dense_460/Tensordot/ReadVariableOp~
dense_460/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_460/Tensordot/axes?
dense_460/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_460/Tensordot/free?
dense_460/Tensordot/ShapeShapeconcatenate_173/concat:output:0*
T0*
_output_shapes
:2
dense_460/Tensordot/Shape?
!dense_460/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_460/Tensordot/GatherV2/axis?
dense_460/Tensordot/GatherV2GatherV2"dense_460/Tensordot/Shape:output:0!dense_460/Tensordot/free:output:0*dense_460/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_460/Tensordot/GatherV2?
#dense_460/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_460/Tensordot/GatherV2_1/axis?
dense_460/Tensordot/GatherV2_1GatherV2"dense_460/Tensordot/Shape:output:0!dense_460/Tensordot/axes:output:0,dense_460/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_460/Tensordot/GatherV2_1?
dense_460/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_460/Tensordot/Const?
dense_460/Tensordot/ProdProd%dense_460/Tensordot/GatherV2:output:0"dense_460/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_460/Tensordot/Prod?
dense_460/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_460/Tensordot/Const_1?
dense_460/Tensordot/Prod_1Prod'dense_460/Tensordot/GatherV2_1:output:0$dense_460/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_460/Tensordot/Prod_1?
dense_460/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_460/Tensordot/concat/axis?
dense_460/Tensordot/concatConcatV2!dense_460/Tensordot/free:output:0!dense_460/Tensordot/axes:output:0(dense_460/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_460/Tensordot/concat?
dense_460/Tensordot/stackPack!dense_460/Tensordot/Prod:output:0#dense_460/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_460/Tensordot/stack?
dense_460/Tensordot/transpose	Transposeconcatenate_173/concat:output:0#dense_460/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_460/Tensordot/transpose?
dense_460/Tensordot/ReshapeReshape!dense_460/Tensordot/transpose:y:0"dense_460/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_460/Tensordot/Reshape?
dense_460/Tensordot/MatMulMatMul$dense_460/Tensordot/Reshape:output:0*dense_460/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_460/Tensordot/MatMul?
dense_460/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_460/Tensordot/Const_2?
!dense_460/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_460/Tensordot/concat_1/axis?
dense_460/Tensordot/concat_1ConcatV2%dense_460/Tensordot/GatherV2:output:0$dense_460/Tensordot/Const_2:output:0*dense_460/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_460/Tensordot/concat_1?
dense_460/TensordotReshape$dense_460/Tensordot/MatMul:product:0%dense_460/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
dense_460/Tensordot?
 dense_460/BiasAdd/ReadVariableOpReadVariableOp)dense_460_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_460/BiasAdd/ReadVariableOp?
dense_460/BiasAddAdddense_460/Tensordot:output:0(dense_460/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
dense_460/BiasAddu
dense_460/ReluReludense_460/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
dense_460/Relu?
tf_op_layer_AddV2_115/AddV2_115AddV2color_law/Tensordot:output:08tf_op_layer_strided_slice_465/strided_slice_465:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2!
tf_op_layer_AddV2_115/AddV2_115?
"dense_461/Tensordot/ReadVariableOpReadVariableOp+dense_461_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"dense_461/Tensordot/ReadVariableOp~
dense_461/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_461/Tensordot/axes?
dense_461/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_461/Tensordot/free?
dense_461/Tensordot/ShapeShapedense_460/Relu:activations:0*
T0*
_output_shapes
:2
dense_461/Tensordot/Shape?
!dense_461/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_461/Tensordot/GatherV2/axis?
dense_461/Tensordot/GatherV2GatherV2"dense_461/Tensordot/Shape:output:0!dense_461/Tensordot/free:output:0*dense_461/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_461/Tensordot/GatherV2?
#dense_461/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_461/Tensordot/GatherV2_1/axis?
dense_461/Tensordot/GatherV2_1GatherV2"dense_461/Tensordot/Shape:output:0!dense_461/Tensordot/axes:output:0,dense_461/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_461/Tensordot/GatherV2_1?
dense_461/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_461/Tensordot/Const?
dense_461/Tensordot/ProdProd%dense_461/Tensordot/GatherV2:output:0"dense_461/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_461/Tensordot/Prod?
dense_461/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_461/Tensordot/Const_1?
dense_461/Tensordot/Prod_1Prod'dense_461/Tensordot/GatherV2_1:output:0$dense_461/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_461/Tensordot/Prod_1?
dense_461/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_461/Tensordot/concat/axis?
dense_461/Tensordot/concatConcatV2!dense_461/Tensordot/free:output:0!dense_461/Tensordot/axes:output:0(dense_461/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_461/Tensordot/concat?
dense_461/Tensordot/stackPack!dense_461/Tensordot/Prod:output:0#dense_461/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_461/Tensordot/stack?
dense_461/Tensordot/transpose	Transposedense_460/Relu:activations:0#dense_461/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2
dense_461/Tensordot/transpose?
dense_461/Tensordot/ReshapeReshape!dense_461/Tensordot/transpose:y:0"dense_461/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_461/Tensordot/Reshape?
dense_461/Tensordot/MatMulMatMul$dense_461/Tensordot/Reshape:output:0*dense_461/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_461/Tensordot/MatMul?
dense_461/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_461/Tensordot/Const_2?
!dense_461/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_461/Tensordot/concat_1/axis?
dense_461/Tensordot/concat_1ConcatV2%dense_461/Tensordot/GatherV2:output:0$dense_461/Tensordot/Const_2:output:0*dense_461/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_461/Tensordot/concat_1?
dense_461/TensordotReshape$dense_461/Tensordot/MatMul:product:0%dense_461/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_461/Tensordot?
 dense_461/BiasAdd/ReadVariableOpReadVariableOp)dense_461_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_461/BiasAdd/ReadVariableOp?
dense_461/BiasAddAdddense_461/Tensordot:output:0(dense_461/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_461/BiasAddv
dense_461/ReluReludense_461/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_461/Relu?
tf_op_layer_Mul_356/Mul_356/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
tf_op_layer_Mul_356/Mul_356/x?
tf_op_layer_Mul_356/Mul_356Mul&tf_op_layer_Mul_356/Mul_356/x:output:0#tf_op_layer_AddV2_115/AddV2_115:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_356/Mul_356?
"dense_462/Tensordot/ReadVariableOpReadVariableOp+dense_462_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_462/Tensordot/ReadVariableOp~
dense_462/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_462/Tensordot/axes?
dense_462/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_462/Tensordot/free?
dense_462/Tensordot/ShapeShapedense_461/Relu:activations:0*
T0*
_output_shapes
:2
dense_462/Tensordot/Shape?
!dense_462/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_462/Tensordot/GatherV2/axis?
dense_462/Tensordot/GatherV2GatherV2"dense_462/Tensordot/Shape:output:0!dense_462/Tensordot/free:output:0*dense_462/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_462/Tensordot/GatherV2?
#dense_462/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_462/Tensordot/GatherV2_1/axis?
dense_462/Tensordot/GatherV2_1GatherV2"dense_462/Tensordot/Shape:output:0!dense_462/Tensordot/axes:output:0,dense_462/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_462/Tensordot/GatherV2_1?
dense_462/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_462/Tensordot/Const?
dense_462/Tensordot/ProdProd%dense_462/Tensordot/GatherV2:output:0"dense_462/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_462/Tensordot/Prod?
dense_462/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_462/Tensordot/Const_1?
dense_462/Tensordot/Prod_1Prod'dense_462/Tensordot/GatherV2_1:output:0$dense_462/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_462/Tensordot/Prod_1?
dense_462/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_462/Tensordot/concat/axis?
dense_462/Tensordot/concatConcatV2!dense_462/Tensordot/free:output:0!dense_462/Tensordot/axes:output:0(dense_462/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_462/Tensordot/concat?
dense_462/Tensordot/stackPack!dense_462/Tensordot/Prod:output:0#dense_462/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_462/Tensordot/stack?
dense_462/Tensordot/transpose	Transposedense_461/Relu:activations:0#dense_462/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_462/Tensordot/transpose?
dense_462/Tensordot/ReshapeReshape!dense_462/Tensordot/transpose:y:0"dense_462/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_462/Tensordot/Reshape?
dense_462/Tensordot/MatMulMatMul$dense_462/Tensordot/Reshape:output:0*dense_462/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_462/Tensordot/MatMul?
dense_462/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_462/Tensordot/Const_2?
!dense_462/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_462/Tensordot/concat_1/axis?
dense_462/Tensordot/concat_1ConcatV2%dense_462/Tensordot/GatherV2:output:0$dense_462/Tensordot/Const_2:output:0*dense_462/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_462/Tensordot/concat_1?
dense_462/TensordotReshape$dense_462/Tensordot/MatMul:product:0%dense_462/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_462/Tensordot?
 dense_462/BiasAdd/ReadVariableOpReadVariableOp)dense_462_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_462/BiasAdd/ReadVariableOp?
dense_462/BiasAddAdddense_462/Tensordot:output:0(dense_462/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_462/BiasAddv
dense_462/ReluReludense_462/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_462/Relu?
"dense_463/Tensordot/ReadVariableOpReadVariableOp+dense_463_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_463/Tensordot/ReadVariableOp~
dense_463/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_463/Tensordot/axes?
dense_463/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_463/Tensordot/free?
dense_463/Tensordot/ShapeShapedense_462/Relu:activations:0*
T0*
_output_shapes
:2
dense_463/Tensordot/Shape?
!dense_463/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_463/Tensordot/GatherV2/axis?
dense_463/Tensordot/GatherV2GatherV2"dense_463/Tensordot/Shape:output:0!dense_463/Tensordot/free:output:0*dense_463/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_463/Tensordot/GatherV2?
#dense_463/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_463/Tensordot/GatherV2_1/axis?
dense_463/Tensordot/GatherV2_1GatherV2"dense_463/Tensordot/Shape:output:0!dense_463/Tensordot/axes:output:0,dense_463/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_463/Tensordot/GatherV2_1?
dense_463/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_463/Tensordot/Const?
dense_463/Tensordot/ProdProd%dense_463/Tensordot/GatherV2:output:0"dense_463/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_463/Tensordot/Prod?
dense_463/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_463/Tensordot/Const_1?
dense_463/Tensordot/Prod_1Prod'dense_463/Tensordot/GatherV2_1:output:0$dense_463/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_463/Tensordot/Prod_1?
dense_463/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_463/Tensordot/concat/axis?
dense_463/Tensordot/concatConcatV2!dense_463/Tensordot/free:output:0!dense_463/Tensordot/axes:output:0(dense_463/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_463/Tensordot/concat?
dense_463/Tensordot/stackPack!dense_463/Tensordot/Prod:output:0#dense_463/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_463/Tensordot/stack?
dense_463/Tensordot/transpose	Transposedense_462/Relu:activations:0#dense_463/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_463/Tensordot/transpose?
dense_463/Tensordot/ReshapeReshape!dense_463/Tensordot/transpose:y:0"dense_463/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_463/Tensordot/Reshape?
dense_463/Tensordot/MatMulMatMul$dense_463/Tensordot/Reshape:output:0*dense_463/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_463/Tensordot/MatMul?
dense_463/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_463/Tensordot/Const_2?
!dense_463/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_463/Tensordot/concat_1/axis?
dense_463/Tensordot/concat_1ConcatV2%dense_463/Tensordot/GatherV2:output:0$dense_463/Tensordot/Const_2:output:0*dense_463/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_463/Tensordot/concat_1?
dense_463/TensordotReshape$dense_463/Tensordot/MatMul:product:0%dense_463/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_463/Tensordot?
 dense_463/BiasAdd/ReadVariableOpReadVariableOp)dense_463_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_463/BiasAdd/ReadVariableOp?
dense_463/BiasAddAdddense_463/Tensordot:output:0(dense_463/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_463/BiasAdd
tf_op_layer_Pow_57/Pow_57/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
tf_op_layer_Pow_57/Pow_57/x?
tf_op_layer_Pow_57/Pow_57Pow$tf_op_layer_Pow_57/Pow_57/x:output:0tf_op_layer_Mul_356/Mul_356:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Pow_57/Pow_57?
tf_op_layer_Mul_357/Mul_357Muldense_463/BiasAdd:z:0tf_op_layer_Pow_57/Pow_57:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_357/Mul_357?
tf_op_layer_Relu_53/Relu_53Relutf_op_layer_Mul_357/Mul_357:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Relu_53/Relu_53?
+tf_op_layer_Max_61/Max_61/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+tf_op_layer_Max_61/Max_61/reduction_indices?
tf_op_layer_Max_61/Max_61Maxinputs_24tf_op_layer_Max_61/Max_61/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
tf_op_layer_Max_61/Max_61?
tf_op_layer_Mul_358/Mul_358Mul)tf_op_layer_Relu_53/Relu_53:activations:0"tf_op_layer_Max_61/Max_61:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_358/Mul_358x
IdentityIdentitytf_op_layer_Mul_358/Mul_358:z:0*
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
?
?
E__inference_dense_460_layer_call_and_return_conditional_losses_457810

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
k
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_457964

inputs
identity[
	Mul_356/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
	Mul_356/x{
Mul_356MulMul_356/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_356d
IdentityIdentityMul_356:z:0*
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
 
_user_specified_nameinputs"?L
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
	input_2327
serving_default_input_232:0????????? ?
G
latent_params6
serving_default_latent_params:0?????????L
tf_op_layer_Mul_3585
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_model??{"class_name": "Model", "name": "model_115", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_115", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}, "name": "latent_params", "inbound_nodes": []}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector_57", "trainable": true, "dtype": "float32", "n": 32}, "name": "repeat_vector_57", "inbound_nodes": [[["latent_params", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}, "name": "conditional_params", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_464", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_464", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_464/begin", "strided_slice_464/end", "strided_slice_464/strides"], "attr": {"new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_464", "inbound_nodes": [[["repeat_vector_57", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_467", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_467", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_467/begin", "strided_slice_467/end", "strided_slice_467/strides"], "attr": {"new_axis_mask": {"i": "0"}, "end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_467", "inbound_nodes": [[["repeat_vector_57", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_114", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_114", "op": "AddV2", "input": ["conditional_params_60", "strided_slice_464"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_114", "inbound_nodes": [[["conditional_params", 0, 0, {}], ["tf_op_layer_strided_slice_464", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_173", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_173", "inbound_nodes": [[["tf_op_layer_strided_slice_467", 0, 0, {}], ["tf_op_layer_AddV2_114", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_466", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_466", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_466/begin", "strided_slice_466/end", "strided_slice_466/strides"], "attr": {"end_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_466", "inbound_nodes": [[["repeat_vector_57", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_460", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_460", "inbound_nodes": [[["concatenate_173", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "color_law", "inbound_nodes": [[["tf_op_layer_strided_slice_466", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_465", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_465", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_465/begin", "strided_slice_465/end", "strided_slice_465/strides"], "attr": {"begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_465", "inbound_nodes": [[["repeat_vector_57", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_461", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_461", "inbound_nodes": [[["dense_460", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_115", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_115", "op": "AddV2", "input": ["color_law_60/Identity", "strided_slice_465"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_115", "inbound_nodes": [[["color_law", 0, 0, {}], ["tf_op_layer_strided_slice_465", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_462", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_462", "inbound_nodes": [[["dense_461", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_356", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_356", "op": "Mul", "input": ["Mul_356/x", "AddV2_115"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}, "name": "tf_op_layer_Mul_356", "inbound_nodes": [[["tf_op_layer_AddV2_115", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_463", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_463", "inbound_nodes": [[["dense_462", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow_57", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_57", "op": "Pow", "input": ["Pow_57/x", "Mul_356"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}, "name": "tf_op_layer_Pow_57", "inbound_nodes": [[["tf_op_layer_Mul_356", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_357", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_357", "op": "Mul", "input": ["dense_463/Identity", "Pow_57"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_357", "inbound_nodes": [[["dense_463", 0, 0, {}], ["tf_op_layer_Pow_57", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_232"}, "name": "input_232", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_53", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_53", "op": "Relu", "input": ["Mul_357"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Relu_53", "inbound_nodes": [[["tf_op_layer_Mul_357", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Max_61", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_61", "op": "Max", "input": ["input_232", "Max_61/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Max_61", "inbound_nodes": [[["input_232", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_358", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_358", "op": "Mul", "input": ["Relu_53", "Max_61"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_358", "inbound_nodes": [[["tf_op_layer_Relu_53", 0, 0, {}], ["tf_op_layer_Max_61", 0, 0, {}]]]}], "input_layers": [["latent_params", 0, 0], ["conditional_params", 0, 0], ["input_232", 0, 0]], "output_layers": [["tf_op_layer_Mul_358", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 6]}, {"class_name": "TensorShape", "items": [null, 32, 1]}, {"class_name": "TensorShape", "items": [null, 32, 288]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_115", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}, "name": "latent_params", "inbound_nodes": []}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector_57", "trainable": true, "dtype": "float32", "n": 32}, "name": "repeat_vector_57", "inbound_nodes": [[["latent_params", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}, "name": "conditional_params", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_464", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_464", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_464/begin", "strided_slice_464/end", "strided_slice_464/strides"], "attr": {"new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_464", "inbound_nodes": [[["repeat_vector_57", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_467", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_467", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_467/begin", "strided_slice_467/end", "strided_slice_467/strides"], "attr": {"new_axis_mask": {"i": "0"}, "end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_467", "inbound_nodes": [[["repeat_vector_57", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_114", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_114", "op": "AddV2", "input": ["conditional_params_60", "strided_slice_464"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_114", "inbound_nodes": [[["conditional_params", 0, 0, {}], ["tf_op_layer_strided_slice_464", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_173", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_173", "inbound_nodes": [[["tf_op_layer_strided_slice_467", 0, 0, {}], ["tf_op_layer_AddV2_114", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_466", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_466", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_466/begin", "strided_slice_466/end", "strided_slice_466/strides"], "attr": {"end_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_466", "inbound_nodes": [[["repeat_vector_57", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_460", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_460", "inbound_nodes": [[["concatenate_173", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "color_law", "inbound_nodes": [[["tf_op_layer_strided_slice_466", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_465", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_465", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_465/begin", "strided_slice_465/end", "strided_slice_465/strides"], "attr": {"begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_465", "inbound_nodes": [[["repeat_vector_57", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_461", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_461", "inbound_nodes": [[["dense_460", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_115", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_115", "op": "AddV2", "input": ["color_law_60/Identity", "strided_slice_465"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_115", "inbound_nodes": [[["color_law", 0, 0, {}], ["tf_op_layer_strided_slice_465", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_462", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_462", "inbound_nodes": [[["dense_461", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_356", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_356", "op": "Mul", "input": ["Mul_356/x", "AddV2_115"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}, "name": "tf_op_layer_Mul_356", "inbound_nodes": [[["tf_op_layer_AddV2_115", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_463", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_463", "inbound_nodes": [[["dense_462", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow_57", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_57", "op": "Pow", "input": ["Pow_57/x", "Mul_356"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}, "name": "tf_op_layer_Pow_57", "inbound_nodes": [[["tf_op_layer_Mul_356", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_357", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_357", "op": "Mul", "input": ["dense_463/Identity", "Pow_57"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_357", "inbound_nodes": [[["dense_463", 0, 0, {}], ["tf_op_layer_Pow_57", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_232"}, "name": "input_232", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_53", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_53", "op": "Relu", "input": ["Mul_357"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Relu_53", "inbound_nodes": [[["tf_op_layer_Mul_357", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Max_61", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_61", "op": "Max", "input": ["input_232", "Max_61/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Max_61", "inbound_nodes": [[["input_232", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_358", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_358", "op": "Mul", "input": ["Relu_53", "Max_61"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_358", "inbound_nodes": [[["tf_op_layer_Relu_53", 0, 0, {}], ["tf_op_layer_Max_61", 0, 0, {}]]]}], "input_layers": [["latent_params", 0, 0], ["conditional_params", 0, 0], ["input_232", 0, 0]], "output_layers": [["tf_op_layer_Mul_358", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "latent_params", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "RepeatVector", "name": "repeat_vector_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "repeat_vector_57", "trainable": true, "dtype": "float32", "n": 32}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conditional_params", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}}
?
 regularization_losses
!	variables
"trainable_variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_464", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_464", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_464", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_464/begin", "strided_slice_464/end", "strided_slice_464/strides"], "attr": {"new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}}
?
$regularization_losses
%	variables
&trainable_variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_467", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_467", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_467", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_467/begin", "strided_slice_467/end", "strided_slice_467/strides"], "attr": {"new_axis_mask": {"i": "0"}, "end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}}
?
(regularization_losses
)	variables
*trainable_variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_114", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_114", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_114", "op": "AddV2", "input": ["conditional_params_60", "strided_slice_464"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_173", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_173", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 3]}, {"class_name": "TensorShape", "items": [null, 32, 1]}]}
?
0regularization_losses
1	variables
2trainable_variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_466", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_466", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_466", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_466/begin", "strided_slice_466/end", "strided_slice_466/strides"], "attr": {"end_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}}
?

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_460", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_460", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 4]}}
?4

:kernel
;regularization_losses
<	variables
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?2
_tf_keras_layer?2{"class_name": "Dense", "name": "color_law", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 1]}}
?
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_465", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_465", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_465", "op": "StridedSlice", "input": ["repeat_vector_57/Identity", "strided_slice_465/begin", "strided_slice_465/end", "strided_slice_465/strides"], "attr": {"begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}}
?

Ckernel
Dbias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_461", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_461", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32]}}
?
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_115", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_115", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_115", "op": "AddV2", "input": ["color_law_60/Identity", "strided_slice_465"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?

Mkernel
Nbias
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_462", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_462", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 128]}}
?
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_356", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_356", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_356", "op": "Mul", "input": ["Mul_356/x", "AddV2_115"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}}
?

Wkernel
Xbias
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_463", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_463", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 256]}}
?
]regularization_losses
^	variables
_trainable_variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Pow_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Pow_57", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_57", "op": "Pow", "input": ["Pow_57/x", "Mul_356"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}}
?
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_357", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_357", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_357", "op": "Mul", "input": ["dense_463/Identity", "Pow_57"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_232", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_232"}}
?
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Relu_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Relu_53", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_53", "op": "Relu", "input": ["Mul_357"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Max_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Max_61", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_61", "op": "Max", "input": ["input_232", "Max_61/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": -1}}}
?
mregularization_losses
n	variables
otrainable_variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_358", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_358", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_358", "op": "Mul", "input": ["Relu_53", "Max_61"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
 "
trackable_list_wrapper
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
regularization_losses
qnon_trainable_variables
	variables
trainable_variables
rlayer_metrics
smetrics
tlayer_regularization_losses

ulayers
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
vlayer_regularization_losses
regularization_losses
wnon_trainable_variables
	variables
trainable_variables
xlayer_metrics
ymetrics

zlayers
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
{layer_regularization_losses
 regularization_losses
|non_trainable_variables
!	variables
"trainable_variables
}layer_metrics
~metrics

layers
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
 ?layer_regularization_losses
$regularization_losses
?non_trainable_variables
%	variables
&trainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
(regularization_losses
?non_trainable_variables
)	variables
*trainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
,regularization_losses
?non_trainable_variables
-	variables
.trainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
0regularization_losses
?non_trainable_variables
1	variables
2trainable_variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
":  2dense_460/kernel
: 2dense_460/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
 ?layer_regularization_losses
6regularization_losses
?non_trainable_variables
7	variables
8trainable_variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$	?2color_law_60/kernel
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
;regularization_losses
?non_trainable_variables
<	variables
=trainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
?regularization_losses
?non_trainable_variables
@	variables
Atrainable_variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	 ?2dense_461/kernel
:?2dense_461/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Eregularization_losses
?non_trainable_variables
F	variables
Gtrainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
Iregularization_losses
?non_trainable_variables
J	variables
Ktrainable_variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_462/kernel
:?2dense_462/bias
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Oregularization_losses
?non_trainable_variables
P	variables
Qtrainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
Sregularization_losses
?non_trainable_variables
T	variables
Utrainable_variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_463/kernel
:?2dense_463/bias
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Yregularization_losses
?non_trainable_variables
Z	variables
[trainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
]regularization_losses
?non_trainable_variables
^	variables
_trainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
aregularization_losses
?non_trainable_variables
b	variables
ctrainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
eregularization_losses
?non_trainable_variables
f	variables
gtrainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
iregularization_losses
?non_trainable_variables
j	variables
ktrainable_variables
?layer_metrics
?metrics
?layers
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
 ?layer_regularization_losses
mregularization_losses
?non_trainable_variables
n	variables
otrainable_variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
:0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
'
:0"
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
 "
trackable_list_wrapper
?2?
*__inference_model_115_layer_call_fn_457300
*__inference_model_115_layer_call_fn_457232
*__inference_model_115_layer_call_fn_457715
*__inference_model_115_layer_call_fn_457690?
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
!__inference__wrapped_model_456676?
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
	input_232????????? ?
?2?
E__inference_model_115_layer_call_and_return_conditional_losses_457496
E__inference_model_115_layer_call_and_return_conditional_losses_457665
E__inference_model_115_layer_call_and_return_conditional_losses_457120
E__inference_model_115_layer_call_and_return_conditional_losses_457163?
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
1__inference_repeat_vector_57_layer_call_fn_456691?
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
L__inference_repeat_vector_57_layer_call_and_return_conditional_losses_456685?
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
>__inference_tf_op_layer_strided_slice_464_layer_call_fn_457728?
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
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_457723?
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
>__inference_tf_op_layer_strided_slice_467_layer_call_fn_457741?
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
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_457736?
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
6__inference_tf_op_layer_AddV2_114_layer_call_fn_457753?
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
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_457747?
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
0__inference_concatenate_173_layer_call_fn_457766?
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
K__inference_concatenate_173_layer_call_and_return_conditional_losses_457760?
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
>__inference_tf_op_layer_strided_slice_466_layer_call_fn_457779?
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
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_457774?
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
*__inference_dense_460_layer_call_fn_457819?
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
E__inference_dense_460_layer_call_and_return_conditional_losses_457810?
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
*__inference_color_law_layer_call_fn_457853?
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
E__inference_color_law_layer_call_and_return_conditional_losses_457846?
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
>__inference_tf_op_layer_strided_slice_465_layer_call_fn_457866?
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
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_457861?
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
*__inference_dense_461_layer_call_fn_457906?
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
E__inference_dense_461_layer_call_and_return_conditional_losses_457897?
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
6__inference_tf_op_layer_AddV2_115_layer_call_fn_457918?
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
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_457912?
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
*__inference_dense_462_layer_call_fn_457958?
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
E__inference_dense_462_layer_call_and_return_conditional_losses_457949?
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
4__inference_tf_op_layer_Mul_356_layer_call_fn_457969?
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
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_457964?
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
*__inference_dense_463_layer_call_fn_458008?
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
E__inference_dense_463_layer_call_and_return_conditional_losses_457999?
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
3__inference_tf_op_layer_Pow_57_layer_call_fn_458019?
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
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_458014?
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
4__inference_tf_op_layer_Mul_357_layer_call_fn_458031?
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
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_458025?
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
4__inference_tf_op_layer_Relu_53_layer_call_fn_458041?
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
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_458036?
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
3__inference_tf_op_layer_Max_61_layer_call_fn_458052?
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
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_458047?
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
4__inference_tf_op_layer_Mul_358_layer_call_fn_458064?
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
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_458058?
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
$__inference_signature_wrapper_457327conditional_params	input_232latent_params?
!__inference__wrapped_model_456676?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_232????????? ?
? "N?K
I
tf_op_layer_Mul_3582?/
tf_op_layer_Mul_358????????? ??
E__inference_color_law_layer_call_and_return_conditional_losses_457846d:3?0
)?&
$?!
inputs????????? 
? "*?'
 ?
0????????? ?
? ?
*__inference_color_law_layer_call_fn_457853W:3?0
)?&
$?!
inputs????????? 
? "?????????? ??
K__inference_concatenate_173_layer_call_and_return_conditional_losses_457760?b?_
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
0__inference_concatenate_173_layer_call_fn_457766?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? "?????????? ?
E__inference_dense_460_layer_call_and_return_conditional_losses_457810d453?0
)?&
$?!
inputs????????? 
? ")?&
?
0?????????  
? ?
*__inference_dense_460_layer_call_fn_457819W453?0
)?&
$?!
inputs????????? 
? "??????????  ?
E__inference_dense_461_layer_call_and_return_conditional_losses_457897eCD3?0
)?&
$?!
inputs?????????  
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_461_layer_call_fn_457906XCD3?0
)?&
$?!
inputs?????????  
? "?????????? ??
E__inference_dense_462_layer_call_and_return_conditional_losses_457949fMN4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_462_layer_call_fn_457958YMN4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
E__inference_dense_463_layer_call_and_return_conditional_losses_457999fWX4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_463_layer_call_fn_458008YWX4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
E__inference_model_115_layer_call_and_return_conditional_losses_457120?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_232????????? ?
p

 
? "*?'
 ?
0????????? ?
? ?
E__inference_model_115_layer_call_and_return_conditional_losses_457163?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_232????????? ?
p 

 
? "*?'
 ?
0????????? ?
? ?
E__inference_model_115_layer_call_and_return_conditional_losses_457496?	:45CDMNWX???
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
E__inference_model_115_layer_call_and_return_conditional_losses_457665?	:45CDMNWX???
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
*__inference_model_115_layer_call_fn_457232?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_232????????? ?
p

 
? "?????????? ??
*__inference_model_115_layer_call_fn_457300?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_232????????? ?
p 

 
? "?????????? ??
*__inference_model_115_layer_call_fn_457690?	:45CDMNWX???
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
*__inference_model_115_layer_call_fn_457715?	:45CDMNWX???
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
L__inference_repeat_vector_57_layer_call_and_return_conditional_losses_456685n8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0????????? ?????????
? ?
1__inference_repeat_vector_57_layer_call_fn_456691a8?5
.?+
)?&
inputs??????????????????
? "%?"????????? ??????????
$__inference_signature_wrapper_457327?	:45CDMNWX???
? 
???
F
conditional_params0?-
conditional_params????????? 
5
	input_232(?%
	input_232????????? ?
8
latent_params'?$
latent_params?????????"N?K
I
tf_op_layer_Mul_3582?/
tf_op_layer_Mul_358????????? ??
Q__inference_tf_op_layer_AddV2_114_layer_call_and_return_conditional_losses_457747?b?_
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
6__inference_tf_op_layer_AddV2_114_layer_call_fn_457753?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? "?????????? ?
Q__inference_tf_op_layer_AddV2_115_layer_call_and_return_conditional_losses_457912?c?`
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
6__inference_tf_op_layer_AddV2_115_layer_call_fn_457918?c?`
Y?V
T?Q
'?$
inputs/0????????? ?
&?#
inputs/1????????? 
? "?????????? ??
N__inference_tf_op_layer_Max_61_layer_call_and_return_conditional_losses_458047a4?1
*?'
%?"
inputs????????? ?
? ")?&
?
0????????? 
? ?
3__inference_tf_op_layer_Max_61_layer_call_fn_458052T4?1
*?'
%?"
inputs????????? ?
? "?????????? ?
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_457964b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
4__inference_tf_op_layer_Mul_356_layer_call_fn_457969U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
O__inference_tf_op_layer_Mul_357_layer_call_and_return_conditional_losses_458025?d?a
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
4__inference_tf_op_layer_Mul_357_layer_call_fn_458031?d?a
Z?W
U?R
'?$
inputs/0????????? ?
'?$
inputs/1????????? ?
? "?????????? ??
O__inference_tf_op_layer_Mul_358_layer_call_and_return_conditional_losses_458058?c?`
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
4__inference_tf_op_layer_Mul_358_layer_call_fn_458064?c?`
Y?V
T?Q
'?$
inputs/0????????? ?
&?#
inputs/1????????? 
? "?????????? ??
N__inference_tf_op_layer_Pow_57_layer_call_and_return_conditional_losses_458014b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
3__inference_tf_op_layer_Pow_57_layer_call_fn_458019U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
O__inference_tf_op_layer_Relu_53_layer_call_and_return_conditional_losses_458036b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
4__inference_tf_op_layer_Relu_53_layer_call_fn_458041U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
Y__inference_tf_op_layer_strided_slice_464_layer_call_and_return_conditional_losses_457723`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_464_layer_call_fn_457728S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_465_layer_call_and_return_conditional_losses_457861`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_465_layer_call_fn_457866S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_466_layer_call_and_return_conditional_losses_457774`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_466_layer_call_fn_457779S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_467_layer_call_and_return_conditional_losses_457736`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_467_layer_call_fn_457741S3?0
)?&
$?!
inputs????????? 
? "?????????? 