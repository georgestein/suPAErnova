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
dense_444/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_444/kernel
u
$dense_444/kernel/Read/ReadVariableOpReadVariableOpdense_444/kernel*
_output_shapes

: *
dtype0
t
dense_444/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_444/bias
m
"dense_444/bias/Read/ReadVariableOpReadVariableOpdense_444/bias*
_output_shapes
: *
dtype0
?
color_law_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_namecolor_law_58/kernel
|
'color_law_58/kernel/Read/ReadVariableOpReadVariableOpcolor_law_58/kernel*
_output_shapes
:	?*
dtype0
}
dense_445/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*!
shared_namedense_445/kernel
v
$dense_445/kernel/Read/ReadVariableOpReadVariableOpdense_445/kernel*
_output_shapes
:	 ?*
dtype0
u
dense_445/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_445/bias
n
"dense_445/bias/Read/ReadVariableOpReadVariableOpdense_445/bias*
_output_shapes	
:?*
dtype0
~
dense_446/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_446/kernel
w
$dense_446/kernel/Read/ReadVariableOpReadVariableOpdense_446/kernel* 
_output_shapes
:
??*
dtype0
u
dense_446/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_446/bias
n
"dense_446/bias/Read/ReadVariableOpReadVariableOpdense_446/bias*
_output_shapes	
:?*
dtype0
~
dense_447/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_447/kernel
w
$dense_447/kernel/Read/ReadVariableOpReadVariableOpdense_447/kernel* 
_output_shapes
:
??*
dtype0
u
dense_447/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_447/bias
n
"dense_447/bias/Read/ReadVariableOpReadVariableOpdense_447/bias*
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
 
R
 trainable_variables
!regularization_losses
"	variables
#	keras_api
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
h

4kernel
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
^

:kernel
;trainable_variables
<regularization_losses
=	variables
>	keras_api
R
?trainable_variables
@regularization_losses
A	variables
B	keras_api
h

Ckernel
Dbias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
R
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
h

Mkernel
Nbias
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
R
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
h

Wkernel
Xbias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
R
]trainable_variables
^regularization_losses
_	variables
`	keras_api
R
atrainable_variables
bregularization_losses
c	variables
d	keras_api
 
R
etrainable_variables
fregularization_losses
g	variables
h	keras_api
R
itrainable_variables
jregularization_losses
k	variables
l	keras_api
R
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
8
40
51
C2
D3
M4
N5
W6
X7
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
?
qlayer_regularization_losses
rmetrics
slayer_metrics
trainable_variables
regularization_losses
	variables

tlayers
unon_trainable_variables
 
 
 
 
?
vlayer_regularization_losses
wmetrics
xlayer_metrics
trainable_variables
regularization_losses
	variables

ylayers
znon_trainable_variables
 
 
 
?
{layer_regularization_losses
|metrics
}layer_metrics
 trainable_variables
!regularization_losses
"	variables

~layers
non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
$trainable_variables
%regularization_losses
&	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
(trainable_variables
)regularization_losses
*	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
,trainable_variables
-regularization_losses
.	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
0trainable_variables
1regularization_losses
2	variables
?layers
?non_trainable_variables
\Z
VARIABLE_VALUEdense_444/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_444/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
?
 ?layer_regularization_losses
?metrics
?layer_metrics
6trainable_variables
7regularization_losses
8	variables
?layers
?non_trainable_variables
_]
VARIABLE_VALUEcolor_law_58/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
 

:0
?
 ?layer_regularization_losses
?metrics
?layer_metrics
;trainable_variables
<regularization_losses
=	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
?trainable_variables
@regularization_losses
A	variables
?layers
?non_trainable_variables
\Z
VARIABLE_VALUEdense_445/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_445/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Etrainable_variables
Fregularization_losses
G	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Itrainable_variables
Jregularization_losses
K	variables
?layers
?non_trainable_variables
\Z
VARIABLE_VALUEdense_446/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_446/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
 

M0
N1
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Otrainable_variables
Pregularization_losses
Q	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Strainable_variables
Tregularization_losses
U	variables
?layers
?non_trainable_variables
\Z
VARIABLE_VALUEdense_447/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_447/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
?
 ?layer_regularization_losses
?metrics
?layer_metrics
Ytrainable_variables
Zregularization_losses
[	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
]trainable_variables
^regularization_losses
_	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
atrainable_variables
bregularization_losses
c	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
etrainable_variables
fregularization_losses
g	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
itrainable_variables
jregularization_losses
k	variables
?layers
?non_trainable_variables
 
 
 
?
 ?layer_regularization_losses
?metrics
?layer_metrics
mtrainable_variables
nregularization_losses
o	variables
?layers
?non_trainable_variables
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
?
"serving_default_conditional_paramsPlaceholder*+
_output_shapes
:????????? *
dtype0* 
shape:????????? 
?
serving_default_input_224Placeholder*,
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
StatefulPartitionedCallStatefulPartitionedCall"serving_default_conditional_paramsserving_default_input_224serving_default_latent_paramscolor_law_58/kerneldense_444/kerneldense_444/biasdense_445/kerneldense_445/biasdense_446/kerneldense_446/biasdense_447/kerneldense_447/bias*
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
$__inference_signature_wrapper_451544
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_444/kernel/Read/ReadVariableOp"dense_444/bias/Read/ReadVariableOp'color_law_58/kernel/Read/ReadVariableOp$dense_445/kernel/Read/ReadVariableOp"dense_445/bias/Read/ReadVariableOp$dense_446/kernel/Read/ReadVariableOp"dense_446/bias/Read/ReadVariableOp$dense_447/kernel/Read/ReadVariableOp"dense_447/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_452337
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_444/kerneldense_444/biascolor_law_58/kerneldense_445/kerneldense_445/biasdense_446/kerneldense_446/biasdense_447/kerneldense_447/bias*
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
"__inference__traced_restore_452376??
?
?
*__inference_model_111_layer_call_fn_451932
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
E__inference_model_111_layer_call_and_return_conditional_losses_4514962
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
Z
>__inference_tf_op_layer_strided_slice_450_layer_call_fn_451996

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
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_4509702
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
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_451940

inputs
identity?
strided_slice_448/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_448/begin
strided_slice_448/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_448/end?
strided_slice_448/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_448/strides?
strided_slice_448StridedSliceinputs strided_slice_448/begin:output:0strided_slice_448/end:output:0"strided_slice_448/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_448r
IdentityIdentitystrided_slice_448:output:0*
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
?
k
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_452253

inputs
identityh
Relu_51Reluinputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Relu_51n
IdentityIdentityRelu_51:activations:0*
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
?
j
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_452264

inputs
identity
Max_59/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max_59/reduction_indices?
Max_59Maxinputs!Max_59/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
Max_59g
IdentityIdentityMax_59:output:0*
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
!__inference__wrapped_model_450893
latent_params
conditional_params
	input_2249
5model_111_color_law_tensordot_readvariableop_resource9
5model_111_dense_444_tensordot_readvariableop_resource7
3model_111_dense_444_biasadd_readvariableop_resource9
5model_111_dense_445_tensordot_readvariableop_resource7
3model_111_dense_445_biasadd_readvariableop_resource9
5model_111_dense_446_tensordot_readvariableop_resource7
3model_111_dense_446_biasadd_readvariableop_resource9
5model_111_dense_447_tensordot_readvariableop_resource7
3model_111_dense_447_biasadd_readvariableop_resource
identity??
)model_111/repeat_vector_55/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_111/repeat_vector_55/ExpandDims/dim?
%model_111/repeat_vector_55/ExpandDims
ExpandDimslatent_params2model_111/repeat_vector_55/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_111/repeat_vector_55/ExpandDims?
 model_111/repeat_vector_55/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_111/repeat_vector_55/stack?
model_111/repeat_vector_55/TileTile.model_111/repeat_vector_55/ExpandDims:output:0)model_111/repeat_vector_55/stack:output:0*
T0*+
_output_shapes
:????????? 2!
model_111/repeat_vector_55/Tile?
?model_111/tf_op_layer_strided_slice_448/strided_slice_448/beginConst*
_output_shapes
:*
dtype0*
valueB"        2A
?model_111/tf_op_layer_strided_slice_448/strided_slice_448/begin?
=model_111/tf_op_layer_strided_slice_448/strided_slice_448/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_111/tf_op_layer_strided_slice_448/strided_slice_448/end?
Amodel_111/tf_op_layer_strided_slice_448/strided_slice_448/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_111/tf_op_layer_strided_slice_448/strided_slice_448/strides?
9model_111/tf_op_layer_strided_slice_448/strided_slice_448StridedSlice(model_111/repeat_vector_55/Tile:output:0Hmodel_111/tf_op_layer_strided_slice_448/strided_slice_448/begin:output:0Fmodel_111/tf_op_layer_strided_slice_448/strided_slice_448/end:output:0Jmodel_111/tf_op_layer_strided_slice_448/strided_slice_448/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_111/tf_op_layer_strided_slice_448/strided_slice_448?
?model_111/tf_op_layer_strided_slice_451/strided_slice_451/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_111/tf_op_layer_strided_slice_451/strided_slice_451/begin?
=model_111/tf_op_layer_strided_slice_451/strided_slice_451/endConst*
_output_shapes
:*
dtype0*
valueB"        2?
=model_111/tf_op_layer_strided_slice_451/strided_slice_451/end?
Amodel_111/tf_op_layer_strided_slice_451/strided_slice_451/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_111/tf_op_layer_strided_slice_451/strided_slice_451/strides?
9model_111/tf_op_layer_strided_slice_451/strided_slice_451StridedSlice(model_111/repeat_vector_55/Tile:output:0Hmodel_111/tf_op_layer_strided_slice_451/strided_slice_451/begin:output:0Fmodel_111/tf_op_layer_strided_slice_451/strided_slice_451/end:output:0Jmodel_111/tf_op_layer_strided_slice_451/strided_slice_451/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2;
9model_111/tf_op_layer_strided_slice_451/strided_slice_451?
)model_111/tf_op_layer_AddV2_110/AddV2_110AddV2conditional_paramsBmodel_111/tf_op_layer_strided_slice_448/strided_slice_448:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2+
)model_111/tf_op_layer_AddV2_110/AddV2_110?
?model_111/tf_op_layer_strided_slice_450/strided_slice_450/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_111/tf_op_layer_strided_slice_450/strided_slice_450/begin?
=model_111/tf_op_layer_strided_slice_450/strided_slice_450/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_111/tf_op_layer_strided_slice_450/strided_slice_450/end?
Amodel_111/tf_op_layer_strided_slice_450/strided_slice_450/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_111/tf_op_layer_strided_slice_450/strided_slice_450/strides?
9model_111/tf_op_layer_strided_slice_450/strided_slice_450StridedSlice(model_111/repeat_vector_55/Tile:output:0Hmodel_111/tf_op_layer_strided_slice_450/strided_slice_450/begin:output:0Fmodel_111/tf_op_layer_strided_slice_450/strided_slice_450/end:output:0Jmodel_111/tf_op_layer_strided_slice_450/strided_slice_450/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_111/tf_op_layer_strided_slice_450/strided_slice_450?
%model_111/concatenate_167/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_111/concatenate_167/concat/axis?
 model_111/concatenate_167/concatConcatV2Bmodel_111/tf_op_layer_strided_slice_451/strided_slice_451:output:0-model_111/tf_op_layer_AddV2_110/AddV2_110:z:0.model_111/concatenate_167/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2"
 model_111/concatenate_167/concat?
,model_111/color_law/Tensordot/ReadVariableOpReadVariableOp5model_111_color_law_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,model_111/color_law/Tensordot/ReadVariableOp?
"model_111/color_law/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_111/color_law/Tensordot/axes?
"model_111/color_law/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_111/color_law/Tensordot/free?
#model_111/color_law/Tensordot/ShapeShapeBmodel_111/tf_op_layer_strided_slice_450/strided_slice_450:output:0*
T0*
_output_shapes
:2%
#model_111/color_law/Tensordot/Shape?
+model_111/color_law/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/color_law/Tensordot/GatherV2/axis?
&model_111/color_law/Tensordot/GatherV2GatherV2,model_111/color_law/Tensordot/Shape:output:0+model_111/color_law/Tensordot/free:output:04model_111/color_law/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_111/color_law/Tensordot/GatherV2?
-model_111/color_law/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_111/color_law/Tensordot/GatherV2_1/axis?
(model_111/color_law/Tensordot/GatherV2_1GatherV2,model_111/color_law/Tensordot/Shape:output:0+model_111/color_law/Tensordot/axes:output:06model_111/color_law/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_111/color_law/Tensordot/GatherV2_1?
#model_111/color_law/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_111/color_law/Tensordot/Const?
"model_111/color_law/Tensordot/ProdProd/model_111/color_law/Tensordot/GatherV2:output:0,model_111/color_law/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_111/color_law/Tensordot/Prod?
%model_111/color_law/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_111/color_law/Tensordot/Const_1?
$model_111/color_law/Tensordot/Prod_1Prod1model_111/color_law/Tensordot/GatherV2_1:output:0.model_111/color_law/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_111/color_law/Tensordot/Prod_1?
)model_111/color_law/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_111/color_law/Tensordot/concat/axis?
$model_111/color_law/Tensordot/concatConcatV2+model_111/color_law/Tensordot/free:output:0+model_111/color_law/Tensordot/axes:output:02model_111/color_law/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_111/color_law/Tensordot/concat?
#model_111/color_law/Tensordot/stackPack+model_111/color_law/Tensordot/Prod:output:0-model_111/color_law/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_111/color_law/Tensordot/stack?
'model_111/color_law/Tensordot/transpose	TransposeBmodel_111/tf_op_layer_strided_slice_450/strided_slice_450:output:0-model_111/color_law/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2)
'model_111/color_law/Tensordot/transpose?
%model_111/color_law/Tensordot/ReshapeReshape+model_111/color_law/Tensordot/transpose:y:0,model_111/color_law/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_111/color_law/Tensordot/Reshape?
$model_111/color_law/Tensordot/MatMulMatMul.model_111/color_law/Tensordot/Reshape:output:04model_111/color_law/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_111/color_law/Tensordot/MatMul?
%model_111/color_law/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_111/color_law/Tensordot/Const_2?
+model_111/color_law/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/color_law/Tensordot/concat_1/axis?
&model_111/color_law/Tensordot/concat_1ConcatV2/model_111/color_law/Tensordot/GatherV2:output:0.model_111/color_law/Tensordot/Const_2:output:04model_111/color_law/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_111/color_law/Tensordot/concat_1?
model_111/color_law/TensordotReshape.model_111/color_law/Tensordot/MatMul:product:0/model_111/color_law/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_111/color_law/Tensordot?
?model_111/tf_op_layer_strided_slice_449/strided_slice_449/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_111/tf_op_layer_strided_slice_449/strided_slice_449/begin?
=model_111/tf_op_layer_strided_slice_449/strided_slice_449/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_111/tf_op_layer_strided_slice_449/strided_slice_449/end?
Amodel_111/tf_op_layer_strided_slice_449/strided_slice_449/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_111/tf_op_layer_strided_slice_449/strided_slice_449/strides?
9model_111/tf_op_layer_strided_slice_449/strided_slice_449StridedSlice(model_111/repeat_vector_55/Tile:output:0Hmodel_111/tf_op_layer_strided_slice_449/strided_slice_449/begin:output:0Fmodel_111/tf_op_layer_strided_slice_449/strided_slice_449/end:output:0Jmodel_111/tf_op_layer_strided_slice_449/strided_slice_449/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2;
9model_111/tf_op_layer_strided_slice_449/strided_slice_449?
,model_111/dense_444/Tensordot/ReadVariableOpReadVariableOp5model_111_dense_444_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02.
,model_111/dense_444/Tensordot/ReadVariableOp?
"model_111/dense_444/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_111/dense_444/Tensordot/axes?
"model_111/dense_444/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_111/dense_444/Tensordot/free?
#model_111/dense_444/Tensordot/ShapeShape)model_111/concatenate_167/concat:output:0*
T0*
_output_shapes
:2%
#model_111/dense_444/Tensordot/Shape?
+model_111/dense_444/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/dense_444/Tensordot/GatherV2/axis?
&model_111/dense_444/Tensordot/GatherV2GatherV2,model_111/dense_444/Tensordot/Shape:output:0+model_111/dense_444/Tensordot/free:output:04model_111/dense_444/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_111/dense_444/Tensordot/GatherV2?
-model_111/dense_444/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_111/dense_444/Tensordot/GatherV2_1/axis?
(model_111/dense_444/Tensordot/GatherV2_1GatherV2,model_111/dense_444/Tensordot/Shape:output:0+model_111/dense_444/Tensordot/axes:output:06model_111/dense_444/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_111/dense_444/Tensordot/GatherV2_1?
#model_111/dense_444/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_111/dense_444/Tensordot/Const?
"model_111/dense_444/Tensordot/ProdProd/model_111/dense_444/Tensordot/GatherV2:output:0,model_111/dense_444/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_111/dense_444/Tensordot/Prod?
%model_111/dense_444/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_111/dense_444/Tensordot/Const_1?
$model_111/dense_444/Tensordot/Prod_1Prod1model_111/dense_444/Tensordot/GatherV2_1:output:0.model_111/dense_444/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_111/dense_444/Tensordot/Prod_1?
)model_111/dense_444/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_111/dense_444/Tensordot/concat/axis?
$model_111/dense_444/Tensordot/concatConcatV2+model_111/dense_444/Tensordot/free:output:0+model_111/dense_444/Tensordot/axes:output:02model_111/dense_444/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_111/dense_444/Tensordot/concat?
#model_111/dense_444/Tensordot/stackPack+model_111/dense_444/Tensordot/Prod:output:0-model_111/dense_444/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_111/dense_444/Tensordot/stack?
'model_111/dense_444/Tensordot/transpose	Transpose)model_111/concatenate_167/concat:output:0-model_111/dense_444/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2)
'model_111/dense_444/Tensordot/transpose?
%model_111/dense_444/Tensordot/ReshapeReshape+model_111/dense_444/Tensordot/transpose:y:0,model_111/dense_444/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_111/dense_444/Tensordot/Reshape?
$model_111/dense_444/Tensordot/MatMulMatMul.model_111/dense_444/Tensordot/Reshape:output:04model_111/dense_444/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$model_111/dense_444/Tensordot/MatMul?
%model_111/dense_444/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_111/dense_444/Tensordot/Const_2?
+model_111/dense_444/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/dense_444/Tensordot/concat_1/axis?
&model_111/dense_444/Tensordot/concat_1ConcatV2/model_111/dense_444/Tensordot/GatherV2:output:0.model_111/dense_444/Tensordot/Const_2:output:04model_111/dense_444/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_111/dense_444/Tensordot/concat_1?
model_111/dense_444/TensordotReshape.model_111/dense_444/Tensordot/MatMul:product:0/model_111/dense_444/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
model_111/dense_444/Tensordot?
*model_111/dense_444/BiasAdd/ReadVariableOpReadVariableOp3model_111_dense_444_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*model_111/dense_444/BiasAdd/ReadVariableOp?
model_111/dense_444/BiasAddAdd&model_111/dense_444/Tensordot:output:02model_111/dense_444/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
model_111/dense_444/BiasAdd?
model_111/dense_444/ReluRelumodel_111/dense_444/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
model_111/dense_444/Relu?
)model_111/tf_op_layer_AddV2_111/AddV2_111AddV2&model_111/color_law/Tensordot:output:0Bmodel_111/tf_op_layer_strided_slice_449/strided_slice_449:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2+
)model_111/tf_op_layer_AddV2_111/AddV2_111?
,model_111/dense_445/Tensordot/ReadVariableOpReadVariableOp5model_111_dense_445_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02.
,model_111/dense_445/Tensordot/ReadVariableOp?
"model_111/dense_445/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_111/dense_445/Tensordot/axes?
"model_111/dense_445/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_111/dense_445/Tensordot/free?
#model_111/dense_445/Tensordot/ShapeShape&model_111/dense_444/Relu:activations:0*
T0*
_output_shapes
:2%
#model_111/dense_445/Tensordot/Shape?
+model_111/dense_445/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/dense_445/Tensordot/GatherV2/axis?
&model_111/dense_445/Tensordot/GatherV2GatherV2,model_111/dense_445/Tensordot/Shape:output:0+model_111/dense_445/Tensordot/free:output:04model_111/dense_445/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_111/dense_445/Tensordot/GatherV2?
-model_111/dense_445/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_111/dense_445/Tensordot/GatherV2_1/axis?
(model_111/dense_445/Tensordot/GatherV2_1GatherV2,model_111/dense_445/Tensordot/Shape:output:0+model_111/dense_445/Tensordot/axes:output:06model_111/dense_445/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_111/dense_445/Tensordot/GatherV2_1?
#model_111/dense_445/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_111/dense_445/Tensordot/Const?
"model_111/dense_445/Tensordot/ProdProd/model_111/dense_445/Tensordot/GatherV2:output:0,model_111/dense_445/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_111/dense_445/Tensordot/Prod?
%model_111/dense_445/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_111/dense_445/Tensordot/Const_1?
$model_111/dense_445/Tensordot/Prod_1Prod1model_111/dense_445/Tensordot/GatherV2_1:output:0.model_111/dense_445/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_111/dense_445/Tensordot/Prod_1?
)model_111/dense_445/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_111/dense_445/Tensordot/concat/axis?
$model_111/dense_445/Tensordot/concatConcatV2+model_111/dense_445/Tensordot/free:output:0+model_111/dense_445/Tensordot/axes:output:02model_111/dense_445/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_111/dense_445/Tensordot/concat?
#model_111/dense_445/Tensordot/stackPack+model_111/dense_445/Tensordot/Prod:output:0-model_111/dense_445/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_111/dense_445/Tensordot/stack?
'model_111/dense_445/Tensordot/transpose	Transpose&model_111/dense_444/Relu:activations:0-model_111/dense_445/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2)
'model_111/dense_445/Tensordot/transpose?
%model_111/dense_445/Tensordot/ReshapeReshape+model_111/dense_445/Tensordot/transpose:y:0,model_111/dense_445/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_111/dense_445/Tensordot/Reshape?
$model_111/dense_445/Tensordot/MatMulMatMul.model_111/dense_445/Tensordot/Reshape:output:04model_111/dense_445/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_111/dense_445/Tensordot/MatMul?
%model_111/dense_445/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_111/dense_445/Tensordot/Const_2?
+model_111/dense_445/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/dense_445/Tensordot/concat_1/axis?
&model_111/dense_445/Tensordot/concat_1ConcatV2/model_111/dense_445/Tensordot/GatherV2:output:0.model_111/dense_445/Tensordot/Const_2:output:04model_111/dense_445/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_111/dense_445/Tensordot/concat_1?
model_111/dense_445/TensordotReshape.model_111/dense_445/Tensordot/MatMul:product:0/model_111/dense_445/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_111/dense_445/Tensordot?
*model_111/dense_445/BiasAdd/ReadVariableOpReadVariableOp3model_111_dense_445_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_111/dense_445/BiasAdd/ReadVariableOp?
model_111/dense_445/BiasAddAdd&model_111/dense_445/Tensordot:output:02model_111/dense_445/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_111/dense_445/BiasAdd?
model_111/dense_445/ReluRelumodel_111/dense_445/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
model_111/dense_445/Relu?
'model_111/tf_op_layer_Mul_330/Mul_330/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2)
'model_111/tf_op_layer_Mul_330/Mul_330/x?
%model_111/tf_op_layer_Mul_330/Mul_330Mul0model_111/tf_op_layer_Mul_330/Mul_330/x:output:0-model_111/tf_op_layer_AddV2_111/AddV2_111:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_111/tf_op_layer_Mul_330/Mul_330?
,model_111/dense_446/Tensordot/ReadVariableOpReadVariableOp5model_111_dense_446_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,model_111/dense_446/Tensordot/ReadVariableOp?
"model_111/dense_446/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_111/dense_446/Tensordot/axes?
"model_111/dense_446/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_111/dense_446/Tensordot/free?
#model_111/dense_446/Tensordot/ShapeShape&model_111/dense_445/Relu:activations:0*
T0*
_output_shapes
:2%
#model_111/dense_446/Tensordot/Shape?
+model_111/dense_446/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/dense_446/Tensordot/GatherV2/axis?
&model_111/dense_446/Tensordot/GatherV2GatherV2,model_111/dense_446/Tensordot/Shape:output:0+model_111/dense_446/Tensordot/free:output:04model_111/dense_446/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_111/dense_446/Tensordot/GatherV2?
-model_111/dense_446/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_111/dense_446/Tensordot/GatherV2_1/axis?
(model_111/dense_446/Tensordot/GatherV2_1GatherV2,model_111/dense_446/Tensordot/Shape:output:0+model_111/dense_446/Tensordot/axes:output:06model_111/dense_446/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_111/dense_446/Tensordot/GatherV2_1?
#model_111/dense_446/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_111/dense_446/Tensordot/Const?
"model_111/dense_446/Tensordot/ProdProd/model_111/dense_446/Tensordot/GatherV2:output:0,model_111/dense_446/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_111/dense_446/Tensordot/Prod?
%model_111/dense_446/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_111/dense_446/Tensordot/Const_1?
$model_111/dense_446/Tensordot/Prod_1Prod1model_111/dense_446/Tensordot/GatherV2_1:output:0.model_111/dense_446/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_111/dense_446/Tensordot/Prod_1?
)model_111/dense_446/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_111/dense_446/Tensordot/concat/axis?
$model_111/dense_446/Tensordot/concatConcatV2+model_111/dense_446/Tensordot/free:output:0+model_111/dense_446/Tensordot/axes:output:02model_111/dense_446/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_111/dense_446/Tensordot/concat?
#model_111/dense_446/Tensordot/stackPack+model_111/dense_446/Tensordot/Prod:output:0-model_111/dense_446/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_111/dense_446/Tensordot/stack?
'model_111/dense_446/Tensordot/transpose	Transpose&model_111/dense_445/Relu:activations:0-model_111/dense_446/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2)
'model_111/dense_446/Tensordot/transpose?
%model_111/dense_446/Tensordot/ReshapeReshape+model_111/dense_446/Tensordot/transpose:y:0,model_111/dense_446/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_111/dense_446/Tensordot/Reshape?
$model_111/dense_446/Tensordot/MatMulMatMul.model_111/dense_446/Tensordot/Reshape:output:04model_111/dense_446/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_111/dense_446/Tensordot/MatMul?
%model_111/dense_446/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_111/dense_446/Tensordot/Const_2?
+model_111/dense_446/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/dense_446/Tensordot/concat_1/axis?
&model_111/dense_446/Tensordot/concat_1ConcatV2/model_111/dense_446/Tensordot/GatherV2:output:0.model_111/dense_446/Tensordot/Const_2:output:04model_111/dense_446/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_111/dense_446/Tensordot/concat_1?
model_111/dense_446/TensordotReshape.model_111/dense_446/Tensordot/MatMul:product:0/model_111/dense_446/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_111/dense_446/Tensordot?
*model_111/dense_446/BiasAdd/ReadVariableOpReadVariableOp3model_111_dense_446_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_111/dense_446/BiasAdd/ReadVariableOp?
model_111/dense_446/BiasAddAdd&model_111/dense_446/Tensordot:output:02model_111/dense_446/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_111/dense_446/BiasAdd?
model_111/dense_446/ReluRelumodel_111/dense_446/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
model_111/dense_446/Relu?
,model_111/dense_447/Tensordot/ReadVariableOpReadVariableOp5model_111_dense_447_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,model_111/dense_447/Tensordot/ReadVariableOp?
"model_111/dense_447/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_111/dense_447/Tensordot/axes?
"model_111/dense_447/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_111/dense_447/Tensordot/free?
#model_111/dense_447/Tensordot/ShapeShape&model_111/dense_446/Relu:activations:0*
T0*
_output_shapes
:2%
#model_111/dense_447/Tensordot/Shape?
+model_111/dense_447/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/dense_447/Tensordot/GatherV2/axis?
&model_111/dense_447/Tensordot/GatherV2GatherV2,model_111/dense_447/Tensordot/Shape:output:0+model_111/dense_447/Tensordot/free:output:04model_111/dense_447/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_111/dense_447/Tensordot/GatherV2?
-model_111/dense_447/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_111/dense_447/Tensordot/GatherV2_1/axis?
(model_111/dense_447/Tensordot/GatherV2_1GatherV2,model_111/dense_447/Tensordot/Shape:output:0+model_111/dense_447/Tensordot/axes:output:06model_111/dense_447/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_111/dense_447/Tensordot/GatherV2_1?
#model_111/dense_447/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_111/dense_447/Tensordot/Const?
"model_111/dense_447/Tensordot/ProdProd/model_111/dense_447/Tensordot/GatherV2:output:0,model_111/dense_447/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_111/dense_447/Tensordot/Prod?
%model_111/dense_447/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_111/dense_447/Tensordot/Const_1?
$model_111/dense_447/Tensordot/Prod_1Prod1model_111/dense_447/Tensordot/GatherV2_1:output:0.model_111/dense_447/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_111/dense_447/Tensordot/Prod_1?
)model_111/dense_447/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_111/dense_447/Tensordot/concat/axis?
$model_111/dense_447/Tensordot/concatConcatV2+model_111/dense_447/Tensordot/free:output:0+model_111/dense_447/Tensordot/axes:output:02model_111/dense_447/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_111/dense_447/Tensordot/concat?
#model_111/dense_447/Tensordot/stackPack+model_111/dense_447/Tensordot/Prod:output:0-model_111/dense_447/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_111/dense_447/Tensordot/stack?
'model_111/dense_447/Tensordot/transpose	Transpose&model_111/dense_446/Relu:activations:0-model_111/dense_447/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2)
'model_111/dense_447/Tensordot/transpose?
%model_111/dense_447/Tensordot/ReshapeReshape+model_111/dense_447/Tensordot/transpose:y:0,model_111/dense_447/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%model_111/dense_447/Tensordot/Reshape?
$model_111/dense_447/Tensordot/MatMulMatMul.model_111/dense_447/Tensordot/Reshape:output:04model_111/dense_447/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_111/dense_447/Tensordot/MatMul?
%model_111/dense_447/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2'
%model_111/dense_447/Tensordot/Const_2?
+model_111/dense_447/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_111/dense_447/Tensordot/concat_1/axis?
&model_111/dense_447/Tensordot/concat_1ConcatV2/model_111/dense_447/Tensordot/GatherV2:output:0.model_111/dense_447/Tensordot/Const_2:output:04model_111/dense_447/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_111/dense_447/Tensordot/concat_1?
model_111/dense_447/TensordotReshape.model_111/dense_447/Tensordot/MatMul:product:0/model_111/dense_447/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
model_111/dense_447/Tensordot?
*model_111/dense_447/BiasAdd/ReadVariableOpReadVariableOp3model_111_dense_447_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_111/dense_447/BiasAdd/ReadVariableOp?
model_111/dense_447/BiasAddAdd&model_111/dense_447/Tensordot:output:02model_111/dense_447/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
model_111/dense_447/BiasAdd?
%model_111/tf_op_layer_Pow_55/Pow_55/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2'
%model_111/tf_op_layer_Pow_55/Pow_55/x?
#model_111/tf_op_layer_Pow_55/Pow_55Pow.model_111/tf_op_layer_Pow_55/Pow_55/x:output:0)model_111/tf_op_layer_Mul_330/Mul_330:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2%
#model_111/tf_op_layer_Pow_55/Pow_55?
%model_111/tf_op_layer_Mul_331/Mul_331Mulmodel_111/dense_447/BiasAdd:z:0'model_111/tf_op_layer_Pow_55/Pow_55:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_111/tf_op_layer_Mul_331/Mul_331?
%model_111/tf_op_layer_Relu_51/Relu_51Relu)model_111/tf_op_layer_Mul_331/Mul_331:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_111/tf_op_layer_Relu_51/Relu_51?
5model_111/tf_op_layer_Max_59/Max_59/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5model_111/tf_op_layer_Max_59/Max_59/reduction_indices?
#model_111/tf_op_layer_Max_59/Max_59Max	input_224>model_111/tf_op_layer_Max_59/Max_59/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2%
#model_111/tf_op_layer_Max_59/Max_59?
%model_111/tf_op_layer_Mul_332/Mul_332Mul3model_111/tf_op_layer_Relu_51/Relu_51:activations:0,model_111/tf_op_layer_Max_59/Max_59:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2'
%model_111/tf_op_layer_Mul_332/Mul_332?
IdentityIdentity)model_111/tf_op_layer_Mul_332/Mul_332:z:0*
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
_user_specified_name	input_224:
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
__inference__traced_save_452337
file_prefix/
+savev2_dense_444_kernel_read_readvariableop-
)savev2_dense_444_bias_read_readvariableop2
.savev2_color_law_58_kernel_read_readvariableop/
+savev2_dense_445_kernel_read_readvariableop-
)savev2_dense_445_bias_read_readvariableop/
+savev2_dense_446_kernel_read_readvariableop-
)savev2_dense_446_bias_read_readvariableop/
+savev2_dense_447_kernel_read_readvariableop-
)savev2_dense_447_bias_read_readvariableop
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
value3B1 B+_temp_b060e5ea2f834906890047838a808a83/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_444_kernel_read_readvariableop)savev2_dense_444_bias_read_readvariableop.savev2_color_law_58_kernel_read_readvariableop+savev2_dense_445_kernel_read_readvariableop)savev2_dense_445_bias_read_readvariableop+savev2_dense_446_kernel_read_readvariableop)savev2_dense_446_bias_read_readvariableop+savev2_dense_447_kernel_read_readvariableop)savev2_dense_447_bias_read_readvariableop"/device:CPU:0*
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
?
?
*__inference_model_111_layer_call_fn_451517
latent_params
conditional_params
	input_224
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
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_224unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
E__inference_model_111_layer_call_and_return_conditional_losses_4514962
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
_user_specified_name	input_224:
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
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_451285

inputs
inputs_1
identityq
Mul_331Mulinputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_331d
IdentityIdentityMul_331:z:0*
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
?
u
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_452078

inputs
identity?
strided_slice_449/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_449/begin
strided_slice_449/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_449/end?
strided_slice_449/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_449/strides?
strided_slice_449StridedSliceinputs strided_slice_449/begin:output:0strided_slice_449/end:output:0"strided_slice_449/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_449r
IdentityIdentitystrided_slice_449:output:0*
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
\
0__inference_concatenate_167_layer_call_fn_451983
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
K__inference_concatenate_167_layer_call_and_return_conditional_losses_4509852
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
?K
?
E__inference_model_111_layer_call_and_return_conditional_losses_451428

inputs
inputs_1
inputs_2
color_law_451396
dense_444_451400
dense_444_451402
dense_445_451406
dense_445_451408
dense_446_451412
dense_446_451414
dense_447_451417
dense_447_451419
identity??!color_law/StatefulPartitionedCall?!dense_444/StatefulPartitionedCall?!dense_445/StatefulPartitionedCall?!dense_446/StatefulPartitionedCall?!dense_447/StatefulPartitionedCall?
 repeat_vector_55/PartitionedCallPartitionedCallinputs*
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
L__inference_repeat_vector_55_layer_call_and_return_conditional_losses_4509022"
 repeat_vector_55/PartitionedCall?
-tf_op_layer_strided_slice_448/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_4509232/
-tf_op_layer_strided_slice_448/PartitionedCall?
-tf_op_layer_strided_slice_451/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_4509392/
-tf_op_layer_strided_slice_451/PartitionedCall?
%tf_op_layer_AddV2_110/PartitionedCallPartitionedCallinputs_16tf_op_layer_strided_slice_448/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_4509532'
%tf_op_layer_AddV2_110/PartitionedCall?
-tf_op_layer_strided_slice_450/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_4509702/
-tf_op_layer_strided_slice_450/PartitionedCall?
concatenate_167/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_451/PartitionedCall:output:0.tf_op_layer_AddV2_110/PartitionedCall:output:0*
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
K__inference_concatenate_167_layer_call_and_return_conditional_losses_4509852!
concatenate_167/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_450/PartitionedCall:output:0color_law_451396*
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
E__inference_color_law_layer_call_and_return_conditional_losses_4510212#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_449/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_4510412/
-tf_op_layer_strided_slice_449/PartitionedCall?
!dense_444/StatefulPartitionedCallStatefulPartitionedCall(concatenate_167/PartitionedCall:output:0dense_444_451400dense_444_451402*
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
E__inference_dense_444_layer_call_and_return_conditional_losses_4510802#
!dense_444/StatefulPartitionedCall?
%tf_op_layer_AddV2_111/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_449/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_4511022'
%tf_op_layer_AddV2_111/PartitionedCall?
!dense_445/StatefulPartitionedCallStatefulPartitionedCall*dense_444/StatefulPartitionedCall:output:0dense_445_451406dense_445_451408*
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
E__inference_dense_445_layer_call_and_return_conditional_losses_4511422#
!dense_445/StatefulPartitionedCall?
#tf_op_layer_Mul_330/PartitionedCallPartitionedCall.tf_op_layer_AddV2_111/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4511642%
#tf_op_layer_Mul_330/PartitionedCall?
!dense_446/StatefulPartitionedCallStatefulPartitionedCall*dense_445/StatefulPartitionedCall:output:0dense_446_451412dense_446_451414*
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
E__inference_dense_446_layer_call_and_return_conditional_losses_4512032#
!dense_446/StatefulPartitionedCall?
!dense_447/StatefulPartitionedCallStatefulPartitionedCall*dense_446/StatefulPartitionedCall:output:0dense_447_451417dense_447_451419*
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
E__inference_dense_447_layer_call_and_return_conditional_losses_4512492#
!dense_447/StatefulPartitionedCall?
"tf_op_layer_Pow_55/PartitionedCallPartitionedCall,tf_op_layer_Mul_330/PartitionedCall:output:0*
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
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_4512712$
"tf_op_layer_Pow_55/PartitionedCall?
#tf_op_layer_Mul_331/PartitionedCallPartitionedCall*dense_447/StatefulPartitionedCall:output:0+tf_op_layer_Pow_55/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_4512852%
#tf_op_layer_Mul_331/PartitionedCall?
#tf_op_layer_Relu_51/PartitionedCallPartitionedCall,tf_op_layer_Mul_331/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_4512992%
#tf_op_layer_Relu_51/PartitionedCall?
"tf_op_layer_Max_59/PartitionedCallPartitionedCallinputs_2*
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
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_4513132$
"tf_op_layer_Max_59/PartitionedCall?
#tf_op_layer_Mul_332/PartitionedCallPartitionedCall,tf_op_layer_Relu_51/PartitionedCall:output:0+tf_op_layer_Max_59/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_4513272%
#tf_op_layer_Mul_332/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_332/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_444/StatefulPartitionedCall"^dense_445/StatefulPartitionedCall"^dense_446/StatefulPartitionedCall"^dense_447/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_444/StatefulPartitionedCall!dense_444/StatefulPartitionedCall2F
!dense_445/StatefulPartitionedCall!dense_445/StatefulPartitionedCall2F
!dense_446/StatefulPartitionedCall!dense_446/StatefulPartitionedCall2F
!dense_447/StatefulPartitionedCall!dense_447/StatefulPartitionedCall:O K
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
?
k
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_452181

inputs
identity[
	Mul_330/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
	Mul_330/x{
Mul_330MulMul_330/x:output:0inputs*
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
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
j
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_451271

inputs
identityY
Pow_55/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

Pow_55/xx
Pow_55PowPow_55/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2
Pow_55c
IdentityIdentity
Pow_55:z:0*
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
E__inference_dense_447_layer_call_and_return_conditional_losses_451249

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
{
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_451102

inputs
inputs_1
identityw
	AddV2_111AddV2inputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2
	AddV2_111f
IdentityIdentityAddV2_111:z:0*
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
E__inference_dense_445_layer_call_and_return_conditional_losses_451142

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
?
Z
>__inference_tf_op_layer_strided_slice_449_layer_call_fn_452083

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
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_4510412
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
?
O
3__inference_tf_op_layer_Pow_55_layer_call_fn_452236

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
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_4512712
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
?
u
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_450970

inputs
identity?
strided_slice_450/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_450/begin
strided_slice_450/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_450/end?
strided_slice_450/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_450/strides?
strided_slice_450StridedSliceinputs strided_slice_450/begin:output:0strided_slice_450/end:output:0"strided_slice_450/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_450r
IdentityIdentitystrided_slice_450:output:0*
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
$__inference_signature_wrapper_451544
conditional_params
	input_224
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
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_224unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
!__inference__wrapped_model_4508932
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
_user_specified_name	input_224:VR
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
?
?
E__inference_color_law_layer_call_and_return_conditional_losses_451021

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
E__inference_model_111_layer_call_and_return_conditional_losses_451496

inputs
inputs_1
inputs_2
color_law_451464
dense_444_451468
dense_444_451470
dense_445_451474
dense_445_451476
dense_446_451480
dense_446_451482
dense_447_451485
dense_447_451487
identity??!color_law/StatefulPartitionedCall?!dense_444/StatefulPartitionedCall?!dense_445/StatefulPartitionedCall?!dense_446/StatefulPartitionedCall?!dense_447/StatefulPartitionedCall?
 repeat_vector_55/PartitionedCallPartitionedCallinputs*
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
L__inference_repeat_vector_55_layer_call_and_return_conditional_losses_4509022"
 repeat_vector_55/PartitionedCall?
-tf_op_layer_strided_slice_448/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_4509232/
-tf_op_layer_strided_slice_448/PartitionedCall?
-tf_op_layer_strided_slice_451/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_4509392/
-tf_op_layer_strided_slice_451/PartitionedCall?
%tf_op_layer_AddV2_110/PartitionedCallPartitionedCallinputs_16tf_op_layer_strided_slice_448/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_4509532'
%tf_op_layer_AddV2_110/PartitionedCall?
-tf_op_layer_strided_slice_450/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_4509702/
-tf_op_layer_strided_slice_450/PartitionedCall?
concatenate_167/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_451/PartitionedCall:output:0.tf_op_layer_AddV2_110/PartitionedCall:output:0*
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
K__inference_concatenate_167_layer_call_and_return_conditional_losses_4509852!
concatenate_167/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_450/PartitionedCall:output:0color_law_451464*
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
E__inference_color_law_layer_call_and_return_conditional_losses_4510212#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_449/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_4510412/
-tf_op_layer_strided_slice_449/PartitionedCall?
!dense_444/StatefulPartitionedCallStatefulPartitionedCall(concatenate_167/PartitionedCall:output:0dense_444_451468dense_444_451470*
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
E__inference_dense_444_layer_call_and_return_conditional_losses_4510802#
!dense_444/StatefulPartitionedCall?
%tf_op_layer_AddV2_111/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_449/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_4511022'
%tf_op_layer_AddV2_111/PartitionedCall?
!dense_445/StatefulPartitionedCallStatefulPartitionedCall*dense_444/StatefulPartitionedCall:output:0dense_445_451474dense_445_451476*
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
E__inference_dense_445_layer_call_and_return_conditional_losses_4511422#
!dense_445/StatefulPartitionedCall?
#tf_op_layer_Mul_330/PartitionedCallPartitionedCall.tf_op_layer_AddV2_111/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4511642%
#tf_op_layer_Mul_330/PartitionedCall?
!dense_446/StatefulPartitionedCallStatefulPartitionedCall*dense_445/StatefulPartitionedCall:output:0dense_446_451480dense_446_451482*
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
E__inference_dense_446_layer_call_and_return_conditional_losses_4512032#
!dense_446/StatefulPartitionedCall?
!dense_447/StatefulPartitionedCallStatefulPartitionedCall*dense_446/StatefulPartitionedCall:output:0dense_447_451485dense_447_451487*
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
E__inference_dense_447_layer_call_and_return_conditional_losses_4512492#
!dense_447/StatefulPartitionedCall?
"tf_op_layer_Pow_55/PartitionedCallPartitionedCall,tf_op_layer_Mul_330/PartitionedCall:output:0*
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
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_4512712$
"tf_op_layer_Pow_55/PartitionedCall?
#tf_op_layer_Mul_331/PartitionedCallPartitionedCall*dense_447/StatefulPartitionedCall:output:0+tf_op_layer_Pow_55/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_4512852%
#tf_op_layer_Mul_331/PartitionedCall?
#tf_op_layer_Relu_51/PartitionedCallPartitionedCall,tf_op_layer_Mul_331/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_4512992%
#tf_op_layer_Relu_51/PartitionedCall?
"tf_op_layer_Max_59/PartitionedCallPartitionedCallinputs_2*
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
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_4513132$
"tf_op_layer_Max_59/PartitionedCall?
#tf_op_layer_Mul_332/PartitionedCallPartitionedCall,tf_op_layer_Relu_51/PartitionedCall:output:0+tf_op_layer_Max_59/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_4513272%
#tf_op_layer_Mul_332/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_332/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_444/StatefulPartitionedCall"^dense_445/StatefulPartitionedCall"^dense_446/StatefulPartitionedCall"^dense_447/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_444/StatefulPartitionedCall!dense_444/StatefulPartitionedCall2F
!dense_445/StatefulPartitionedCall!dense_445/StatefulPartitionedCall2F
!dense_446/StatefulPartitionedCall!dense_446/StatefulPartitionedCall2F
!dense_447/StatefulPartitionedCall!dense_447/StatefulPartitionedCall:O K
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
?
?
*__inference_model_111_layer_call_fn_451449
latent_params
conditional_params
	input_224
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
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_224unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
E__inference_model_111_layer_call_and_return_conditional_losses_4514282
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
_user_specified_name	input_224:
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
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_451953

inputs
identity?
strided_slice_451/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_451/begin
strided_slice_451/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_451/end?
strided_slice_451/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_451/strides?
strided_slice_451StridedSliceinputs strided_slice_451/begin:output:0strided_slice_451/end:output:0"strided_slice_451/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2
strided_slice_451r
IdentityIdentitystrided_slice_451:output:0*
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
?
j
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_452231

inputs
identityY
Pow_55/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

Pow_55/xx
Pow_55PowPow_55/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2
Pow_55c
IdentityIdentity
Pow_55:z:0*
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
?
*__inference_model_111_layer_call_fn_451907
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
E__inference_model_111_layer_call_and_return_conditional_losses_4514282
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
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_451041

inputs
identity?
strided_slice_449/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_449/begin
strided_slice_449/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_449/end?
strided_slice_449/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_449/strides?
strided_slice_449StridedSliceinputs strided_slice_449/begin:output:0strided_slice_449/end:output:0"strided_slice_449/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_449r
IdentityIdentitystrided_slice_449:output:0*
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
?
E__inference_model_111_layer_call_and_return_conditional_losses_451337
latent_params
conditional_params
	input_224
color_law_451030
dense_444_451091
dense_444_451093
dense_445_451153
dense_445_451155
dense_446_451214
dense_446_451216
dense_447_451260
dense_447_451262
identity??!color_law/StatefulPartitionedCall?!dense_444/StatefulPartitionedCall?!dense_445/StatefulPartitionedCall?!dense_446/StatefulPartitionedCall?!dense_447/StatefulPartitionedCall?
 repeat_vector_55/PartitionedCallPartitionedCalllatent_params*
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
L__inference_repeat_vector_55_layer_call_and_return_conditional_losses_4509022"
 repeat_vector_55/PartitionedCall?
-tf_op_layer_strided_slice_448/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_4509232/
-tf_op_layer_strided_slice_448/PartitionedCall?
-tf_op_layer_strided_slice_451/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_4509392/
-tf_op_layer_strided_slice_451/PartitionedCall?
%tf_op_layer_AddV2_110/PartitionedCallPartitionedCallconditional_params6tf_op_layer_strided_slice_448/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_4509532'
%tf_op_layer_AddV2_110/PartitionedCall?
-tf_op_layer_strided_slice_450/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_4509702/
-tf_op_layer_strided_slice_450/PartitionedCall?
concatenate_167/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_451/PartitionedCall:output:0.tf_op_layer_AddV2_110/PartitionedCall:output:0*
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
K__inference_concatenate_167_layer_call_and_return_conditional_losses_4509852!
concatenate_167/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_450/PartitionedCall:output:0color_law_451030*
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
E__inference_color_law_layer_call_and_return_conditional_losses_4510212#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_449/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_4510412/
-tf_op_layer_strided_slice_449/PartitionedCall?
!dense_444/StatefulPartitionedCallStatefulPartitionedCall(concatenate_167/PartitionedCall:output:0dense_444_451091dense_444_451093*
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
E__inference_dense_444_layer_call_and_return_conditional_losses_4510802#
!dense_444/StatefulPartitionedCall?
%tf_op_layer_AddV2_111/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_449/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_4511022'
%tf_op_layer_AddV2_111/PartitionedCall?
!dense_445/StatefulPartitionedCallStatefulPartitionedCall*dense_444/StatefulPartitionedCall:output:0dense_445_451153dense_445_451155*
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
E__inference_dense_445_layer_call_and_return_conditional_losses_4511422#
!dense_445/StatefulPartitionedCall?
#tf_op_layer_Mul_330/PartitionedCallPartitionedCall.tf_op_layer_AddV2_111/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4511642%
#tf_op_layer_Mul_330/PartitionedCall?
!dense_446/StatefulPartitionedCallStatefulPartitionedCall*dense_445/StatefulPartitionedCall:output:0dense_446_451214dense_446_451216*
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
E__inference_dense_446_layer_call_and_return_conditional_losses_4512032#
!dense_446/StatefulPartitionedCall?
!dense_447/StatefulPartitionedCallStatefulPartitionedCall*dense_446/StatefulPartitionedCall:output:0dense_447_451260dense_447_451262*
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
E__inference_dense_447_layer_call_and_return_conditional_losses_4512492#
!dense_447/StatefulPartitionedCall?
"tf_op_layer_Pow_55/PartitionedCallPartitionedCall,tf_op_layer_Mul_330/PartitionedCall:output:0*
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
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_4512712$
"tf_op_layer_Pow_55/PartitionedCall?
#tf_op_layer_Mul_331/PartitionedCallPartitionedCall*dense_447/StatefulPartitionedCall:output:0+tf_op_layer_Pow_55/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_4512852%
#tf_op_layer_Mul_331/PartitionedCall?
#tf_op_layer_Relu_51/PartitionedCallPartitionedCall,tf_op_layer_Mul_331/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_4512992%
#tf_op_layer_Relu_51/PartitionedCall?
"tf_op_layer_Max_59/PartitionedCallPartitionedCall	input_224*
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
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_4513132$
"tf_op_layer_Max_59/PartitionedCall?
#tf_op_layer_Mul_332/PartitionedCallPartitionedCall,tf_op_layer_Relu_51/PartitionedCall:output:0+tf_op_layer_Max_59/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_4513272%
#tf_op_layer_Mul_332/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_332/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_444/StatefulPartitionedCall"^dense_445/StatefulPartitionedCall"^dense_446/StatefulPartitionedCall"^dense_447/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_444/StatefulPartitionedCall!dense_444/StatefulPartitionedCall2F
!dense_445/StatefulPartitionedCall!dense_445/StatefulPartitionedCall2F
!dense_446/StatefulPartitionedCall!dense_446/StatefulPartitionedCall2F
!dense_447/StatefulPartitionedCall!dense_447/StatefulPartitionedCall:V R
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
_user_specified_name	input_224:
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
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_450923

inputs
identity?
strided_slice_448/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_448/begin
strided_slice_448/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_448/end?
strided_slice_448/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_448/strides?
strided_slice_448StridedSliceinputs strided_slice_448/begin:output:0strided_slice_448/end:output:0"strided_slice_448/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_448r
IdentityIdentitystrided_slice_448:output:0*
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
j
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_451313

inputs
identity
Max_59/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max_59/reduction_indices?
Max_59Maxinputs!Max_59/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
Max_59g
IdentityIdentityMax_59:output:0*
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
b
6__inference_tf_op_layer_AddV2_111_layer_call_fn_452135
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
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_4511022
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
?
{
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_450953

inputs
inputs_1
identityv
	AddV2_110AddV2inputsinputs_1*
T0*
_cloned(*+
_output_shapes
:????????? 2
	AddV2_110e
IdentityIdentityAddV2_110:z:0*
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
?
u
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_451991

inputs
identity?
strided_slice_450/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_450/begin
strided_slice_450/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_450/end?
strided_slice_450/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_450/strides?
strided_slice_450StridedSliceinputs strided_slice_450/begin:output:0strided_slice_450/end:output:0"strided_slice_450/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask2
strided_slice_450r
IdentityIdentitystrided_slice_450:output:0*
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
w
K__inference_concatenate_167_layer_call_and_return_conditional_losses_451977
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
?
}
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_452129
inputs_0
inputs_1
identityy
	AddV2_111AddV2inputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2
	AddV2_111f
IdentityIdentityAddV2_111:z:0*
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
?
?
E__inference_dense_444_layer_call_and_return_conditional_losses_451080

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
?
Z
>__inference_tf_op_layer_strided_slice_451_layer_call_fn_451958

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
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_4509392
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
?.
?
"__inference__traced_restore_452376
file_prefix%
!assignvariableop_dense_444_kernel%
!assignvariableop_1_dense_444_bias*
&assignvariableop_2_color_law_58_kernel'
#assignvariableop_3_dense_445_kernel%
!assignvariableop_4_dense_445_bias'
#assignvariableop_5_dense_446_kernel%
!assignvariableop_6_dense_446_bias'
#assignvariableop_7_dense_447_kernel%
!assignvariableop_8_dense_447_bias
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_444_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_444_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_color_law_58_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_445_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_445_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_446_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_446_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_447_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_447_biasIdentity_8:output:0*
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
?
k
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_451299

inputs
identityh
Relu_51Reluinputs*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Relu_51n
IdentityIdentityRelu_51:activations:0*
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
b
6__inference_tf_op_layer_AddV2_110_layer_call_fn_451970
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
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_4509532
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
?
Z
>__inference_tf_op_layer_strided_slice_448_layer_call_fn_451945

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
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_4509232
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
?

*__inference_dense_444_layer_call_fn_452036

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
E__inference_dense_444_layer_call_and_return_conditional_losses_4510802
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
4__inference_tf_op_layer_Mul_330_layer_call_fn_452186

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
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4511642
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
?K
?
E__inference_model_111_layer_call_and_return_conditional_losses_451380
latent_params
conditional_params
	input_224
color_law_451348
dense_444_451352
dense_444_451354
dense_445_451358
dense_445_451360
dense_446_451364
dense_446_451366
dense_447_451369
dense_447_451371
identity??!color_law/StatefulPartitionedCall?!dense_444/StatefulPartitionedCall?!dense_445/StatefulPartitionedCall?!dense_446/StatefulPartitionedCall?!dense_447/StatefulPartitionedCall?
 repeat_vector_55/PartitionedCallPartitionedCalllatent_params*
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
L__inference_repeat_vector_55_layer_call_and_return_conditional_losses_4509022"
 repeat_vector_55/PartitionedCall?
-tf_op_layer_strided_slice_448/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_4509232/
-tf_op_layer_strided_slice_448/PartitionedCall?
-tf_op_layer_strided_slice_451/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_4509392/
-tf_op_layer_strided_slice_451/PartitionedCall?
%tf_op_layer_AddV2_110/PartitionedCallPartitionedCallconditional_params6tf_op_layer_strided_slice_448/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_4509532'
%tf_op_layer_AddV2_110/PartitionedCall?
-tf_op_layer_strided_slice_450/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_4509702/
-tf_op_layer_strided_slice_450/PartitionedCall?
concatenate_167/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_451/PartitionedCall:output:0.tf_op_layer_AddV2_110/PartitionedCall:output:0*
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
K__inference_concatenate_167_layer_call_and_return_conditional_losses_4509852!
concatenate_167/PartitionedCall?
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_450/PartitionedCall:output:0color_law_451348*
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
E__inference_color_law_layer_call_and_return_conditional_losses_4510212#
!color_law/StatefulPartitionedCall?
-tf_op_layer_strided_slice_449/PartitionedCallPartitionedCall)repeat_vector_55/PartitionedCall:output:0*
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
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_4510412/
-tf_op_layer_strided_slice_449/PartitionedCall?
!dense_444/StatefulPartitionedCallStatefulPartitionedCall(concatenate_167/PartitionedCall:output:0dense_444_451352dense_444_451354*
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
E__inference_dense_444_layer_call_and_return_conditional_losses_4510802#
!dense_444/StatefulPartitionedCall?
%tf_op_layer_AddV2_111/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_449/PartitionedCall:output:0*
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
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_4511022'
%tf_op_layer_AddV2_111/PartitionedCall?
!dense_445/StatefulPartitionedCallStatefulPartitionedCall*dense_444/StatefulPartitionedCall:output:0dense_445_451358dense_445_451360*
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
E__inference_dense_445_layer_call_and_return_conditional_losses_4511422#
!dense_445/StatefulPartitionedCall?
#tf_op_layer_Mul_330/PartitionedCallPartitionedCall.tf_op_layer_AddV2_111/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_4511642%
#tf_op_layer_Mul_330/PartitionedCall?
!dense_446/StatefulPartitionedCallStatefulPartitionedCall*dense_445/StatefulPartitionedCall:output:0dense_446_451364dense_446_451366*
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
E__inference_dense_446_layer_call_and_return_conditional_losses_4512032#
!dense_446/StatefulPartitionedCall?
!dense_447/StatefulPartitionedCallStatefulPartitionedCall*dense_446/StatefulPartitionedCall:output:0dense_447_451369dense_447_451371*
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
E__inference_dense_447_layer_call_and_return_conditional_losses_4512492#
!dense_447/StatefulPartitionedCall?
"tf_op_layer_Pow_55/PartitionedCallPartitionedCall,tf_op_layer_Mul_330/PartitionedCall:output:0*
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
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_4512712$
"tf_op_layer_Pow_55/PartitionedCall?
#tf_op_layer_Mul_331/PartitionedCallPartitionedCall*dense_447/StatefulPartitionedCall:output:0+tf_op_layer_Pow_55/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_4512852%
#tf_op_layer_Mul_331/PartitionedCall?
#tf_op_layer_Relu_51/PartitionedCallPartitionedCall,tf_op_layer_Mul_331/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_4512992%
#tf_op_layer_Relu_51/PartitionedCall?
"tf_op_layer_Max_59/PartitionedCallPartitionedCall	input_224*
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
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_4513132$
"tf_op_layer_Max_59/PartitionedCall?
#tf_op_layer_Mul_332/PartitionedCallPartitionedCall,tf_op_layer_Relu_51/PartitionedCall:output:0+tf_op_layer_Max_59/PartitionedCall:output:0*
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
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_4513272%
#tf_op_layer_Mul_332/PartitionedCall?
IdentityIdentity,tf_op_layer_Mul_332/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_444/StatefulPartitionedCall"^dense_445/StatefulPartitionedCall"^dense_446/StatefulPartitionedCall"^dense_447/StatefulPartitionedCall*
T0*,
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:?????????:????????? :????????? ?:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_444/StatefulPartitionedCall!dense_444/StatefulPartitionedCall2F
!dense_445/StatefulPartitionedCall!dense_445/StatefulPartitionedCall2F
!dense_446/StatefulPartitionedCall!dense_446/StatefulPartitionedCall2F
!dense_447/StatefulPartitionedCall!dense_447/StatefulPartitionedCall:V R
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
_user_specified_name	input_224:
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
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_451964
inputs_0
inputs_1
identityx
	AddV2_110AddV2inputs_0inputs_1*
T0*
_cloned(*+
_output_shapes
:????????? 2
	AddV2_110e
IdentityIdentityAddV2_110:z:0*
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
?
`
4__inference_tf_op_layer_Mul_331_layer_call_fn_452248
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
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_4512852
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
?
?
E__inference_dense_445_layer_call_and_return_conditional_losses_452114

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
??
?
E__inference_model_111_layer_call_and_return_conditional_losses_451882
inputs_0
inputs_1
inputs_2/
+color_law_tensordot_readvariableop_resource/
+dense_444_tensordot_readvariableop_resource-
)dense_444_biasadd_readvariableop_resource/
+dense_445_tensordot_readvariableop_resource-
)dense_445_biasadd_readvariableop_resource/
+dense_446_tensordot_readvariableop_resource-
)dense_446_biasadd_readvariableop_resource/
+dense_447_tensordot_readvariableop_resource-
)dense_447_biasadd_readvariableop_resource
identity??
repeat_vector_55/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_55/ExpandDims/dim?
repeat_vector_55/ExpandDims
ExpandDimsinputs_0(repeat_vector_55/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
repeat_vector_55/ExpandDims?
repeat_vector_55/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
repeat_vector_55/stack?
repeat_vector_55/TileTile$repeat_vector_55/ExpandDims:output:0repeat_vector_55/stack:output:0*
T0*+
_output_shapes
:????????? 2
repeat_vector_55/Tile?
5tf_op_layer_strided_slice_448/strided_slice_448/beginConst*
_output_shapes
:*
dtype0*
valueB"        27
5tf_op_layer_strided_slice_448/strided_slice_448/begin?
3tf_op_layer_strided_slice_448/strided_slice_448/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_448/strided_slice_448/end?
7tf_op_layer_strided_slice_448/strided_slice_448/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_448/strided_slice_448/strides?
/tf_op_layer_strided_slice_448/strided_slice_448StridedSlicerepeat_vector_55/Tile:output:0>tf_op_layer_strided_slice_448/strided_slice_448/begin:output:0<tf_op_layer_strided_slice_448/strided_slice_448/end:output:0@tf_op_layer_strided_slice_448/strided_slice_448/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_448/strided_slice_448?
5tf_op_layer_strided_slice_451/strided_slice_451/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_451/strided_slice_451/begin?
3tf_op_layer_strided_slice_451/strided_slice_451/endConst*
_output_shapes
:*
dtype0*
valueB"        25
3tf_op_layer_strided_slice_451/strided_slice_451/end?
7tf_op_layer_strided_slice_451/strided_slice_451/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_451/strided_slice_451/strides?
/tf_op_layer_strided_slice_451/strided_slice_451StridedSlicerepeat_vector_55/Tile:output:0>tf_op_layer_strided_slice_451/strided_slice_451/begin:output:0<tf_op_layer_strided_slice_451/strided_slice_451/end:output:0@tf_op_layer_strided_slice_451/strided_slice_451/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask21
/tf_op_layer_strided_slice_451/strided_slice_451?
tf_op_layer_AddV2_110/AddV2_110AddV2inputs_18tf_op_layer_strided_slice_448/strided_slice_448:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2!
tf_op_layer_AddV2_110/AddV2_110?
5tf_op_layer_strided_slice_450/strided_slice_450/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_450/strided_slice_450/begin?
3tf_op_layer_strided_slice_450/strided_slice_450/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_450/strided_slice_450/end?
7tf_op_layer_strided_slice_450/strided_slice_450/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_450/strided_slice_450/strides?
/tf_op_layer_strided_slice_450/strided_slice_450StridedSlicerepeat_vector_55/Tile:output:0>tf_op_layer_strided_slice_450/strided_slice_450/begin:output:0<tf_op_layer_strided_slice_450/strided_slice_450/end:output:0@tf_op_layer_strided_slice_450/strided_slice_450/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_450/strided_slice_450|
concatenate_167/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_167/concat/axis?
concatenate_167/concatConcatV28tf_op_layer_strided_slice_451/strided_slice_451:output:0#tf_op_layer_AddV2_110/AddV2_110:z:0$concatenate_167/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concatenate_167/concat?
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
color_law/Tensordot/ShapeShape8tf_op_layer_strided_slice_450/strided_slice_450:output:0*
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
color_law/Tensordot/transpose	Transpose8tf_op_layer_strided_slice_450/strided_slice_450:output:0#color_law/Tensordot/concat:output:0*
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
5tf_op_layer_strided_slice_449/strided_slice_449/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_449/strided_slice_449/begin?
3tf_op_layer_strided_slice_449/strided_slice_449/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_449/strided_slice_449/end?
7tf_op_layer_strided_slice_449/strided_slice_449/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_449/strided_slice_449/strides?
/tf_op_layer_strided_slice_449/strided_slice_449StridedSlicerepeat_vector_55/Tile:output:0>tf_op_layer_strided_slice_449/strided_slice_449/begin:output:0<tf_op_layer_strided_slice_449/strided_slice_449/end:output:0@tf_op_layer_strided_slice_449/strided_slice_449/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_449/strided_slice_449?
"dense_444/Tensordot/ReadVariableOpReadVariableOp+dense_444_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02$
"dense_444/Tensordot/ReadVariableOp~
dense_444/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_444/Tensordot/axes?
dense_444/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_444/Tensordot/free?
dense_444/Tensordot/ShapeShapeconcatenate_167/concat:output:0*
T0*
_output_shapes
:2
dense_444/Tensordot/Shape?
!dense_444/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_444/Tensordot/GatherV2/axis?
dense_444/Tensordot/GatherV2GatherV2"dense_444/Tensordot/Shape:output:0!dense_444/Tensordot/free:output:0*dense_444/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_444/Tensordot/GatherV2?
#dense_444/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_444/Tensordot/GatherV2_1/axis?
dense_444/Tensordot/GatherV2_1GatherV2"dense_444/Tensordot/Shape:output:0!dense_444/Tensordot/axes:output:0,dense_444/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_444/Tensordot/GatherV2_1?
dense_444/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_444/Tensordot/Const?
dense_444/Tensordot/ProdProd%dense_444/Tensordot/GatherV2:output:0"dense_444/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_444/Tensordot/Prod?
dense_444/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_444/Tensordot/Const_1?
dense_444/Tensordot/Prod_1Prod'dense_444/Tensordot/GatherV2_1:output:0$dense_444/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_444/Tensordot/Prod_1?
dense_444/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_444/Tensordot/concat/axis?
dense_444/Tensordot/concatConcatV2!dense_444/Tensordot/free:output:0!dense_444/Tensordot/axes:output:0(dense_444/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_444/Tensordot/concat?
dense_444/Tensordot/stackPack!dense_444/Tensordot/Prod:output:0#dense_444/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_444/Tensordot/stack?
dense_444/Tensordot/transpose	Transposeconcatenate_167/concat:output:0#dense_444/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_444/Tensordot/transpose?
dense_444/Tensordot/ReshapeReshape!dense_444/Tensordot/transpose:y:0"dense_444/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_444/Tensordot/Reshape?
dense_444/Tensordot/MatMulMatMul$dense_444/Tensordot/Reshape:output:0*dense_444/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_444/Tensordot/MatMul?
dense_444/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_444/Tensordot/Const_2?
!dense_444/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_444/Tensordot/concat_1/axis?
dense_444/Tensordot/concat_1ConcatV2%dense_444/Tensordot/GatherV2:output:0$dense_444/Tensordot/Const_2:output:0*dense_444/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_444/Tensordot/concat_1?
dense_444/TensordotReshape$dense_444/Tensordot/MatMul:product:0%dense_444/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
dense_444/Tensordot?
 dense_444/BiasAdd/ReadVariableOpReadVariableOp)dense_444_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_444/BiasAdd/ReadVariableOp?
dense_444/BiasAddAdddense_444/Tensordot:output:0(dense_444/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
dense_444/BiasAddu
dense_444/ReluReludense_444/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
dense_444/Relu?
tf_op_layer_AddV2_111/AddV2_111AddV2color_law/Tensordot:output:08tf_op_layer_strided_slice_449/strided_slice_449:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2!
tf_op_layer_AddV2_111/AddV2_111?
"dense_445/Tensordot/ReadVariableOpReadVariableOp+dense_445_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"dense_445/Tensordot/ReadVariableOp~
dense_445/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_445/Tensordot/axes?
dense_445/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_445/Tensordot/free?
dense_445/Tensordot/ShapeShapedense_444/Relu:activations:0*
T0*
_output_shapes
:2
dense_445/Tensordot/Shape?
!dense_445/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_445/Tensordot/GatherV2/axis?
dense_445/Tensordot/GatherV2GatherV2"dense_445/Tensordot/Shape:output:0!dense_445/Tensordot/free:output:0*dense_445/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_445/Tensordot/GatherV2?
#dense_445/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_445/Tensordot/GatherV2_1/axis?
dense_445/Tensordot/GatherV2_1GatherV2"dense_445/Tensordot/Shape:output:0!dense_445/Tensordot/axes:output:0,dense_445/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_445/Tensordot/GatherV2_1?
dense_445/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_445/Tensordot/Const?
dense_445/Tensordot/ProdProd%dense_445/Tensordot/GatherV2:output:0"dense_445/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_445/Tensordot/Prod?
dense_445/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_445/Tensordot/Const_1?
dense_445/Tensordot/Prod_1Prod'dense_445/Tensordot/GatherV2_1:output:0$dense_445/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_445/Tensordot/Prod_1?
dense_445/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_445/Tensordot/concat/axis?
dense_445/Tensordot/concatConcatV2!dense_445/Tensordot/free:output:0!dense_445/Tensordot/axes:output:0(dense_445/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_445/Tensordot/concat?
dense_445/Tensordot/stackPack!dense_445/Tensordot/Prod:output:0#dense_445/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_445/Tensordot/stack?
dense_445/Tensordot/transpose	Transposedense_444/Relu:activations:0#dense_445/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2
dense_445/Tensordot/transpose?
dense_445/Tensordot/ReshapeReshape!dense_445/Tensordot/transpose:y:0"dense_445/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_445/Tensordot/Reshape?
dense_445/Tensordot/MatMulMatMul$dense_445/Tensordot/Reshape:output:0*dense_445/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_445/Tensordot/MatMul?
dense_445/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_445/Tensordot/Const_2?
!dense_445/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_445/Tensordot/concat_1/axis?
dense_445/Tensordot/concat_1ConcatV2%dense_445/Tensordot/GatherV2:output:0$dense_445/Tensordot/Const_2:output:0*dense_445/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_445/Tensordot/concat_1?
dense_445/TensordotReshape$dense_445/Tensordot/MatMul:product:0%dense_445/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_445/Tensordot?
 dense_445/BiasAdd/ReadVariableOpReadVariableOp)dense_445_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_445/BiasAdd/ReadVariableOp?
dense_445/BiasAddAdddense_445/Tensordot:output:0(dense_445/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_445/BiasAddv
dense_445/ReluReludense_445/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_445/Relu?
tf_op_layer_Mul_330/Mul_330/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
tf_op_layer_Mul_330/Mul_330/x?
tf_op_layer_Mul_330/Mul_330Mul&tf_op_layer_Mul_330/Mul_330/x:output:0#tf_op_layer_AddV2_111/AddV2_111:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_330/Mul_330?
"dense_446/Tensordot/ReadVariableOpReadVariableOp+dense_446_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_446/Tensordot/ReadVariableOp~
dense_446/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_446/Tensordot/axes?
dense_446/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_446/Tensordot/free?
dense_446/Tensordot/ShapeShapedense_445/Relu:activations:0*
T0*
_output_shapes
:2
dense_446/Tensordot/Shape?
!dense_446/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_446/Tensordot/GatherV2/axis?
dense_446/Tensordot/GatherV2GatherV2"dense_446/Tensordot/Shape:output:0!dense_446/Tensordot/free:output:0*dense_446/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_446/Tensordot/GatherV2?
#dense_446/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_446/Tensordot/GatherV2_1/axis?
dense_446/Tensordot/GatherV2_1GatherV2"dense_446/Tensordot/Shape:output:0!dense_446/Tensordot/axes:output:0,dense_446/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_446/Tensordot/GatherV2_1?
dense_446/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_446/Tensordot/Const?
dense_446/Tensordot/ProdProd%dense_446/Tensordot/GatherV2:output:0"dense_446/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_446/Tensordot/Prod?
dense_446/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_446/Tensordot/Const_1?
dense_446/Tensordot/Prod_1Prod'dense_446/Tensordot/GatherV2_1:output:0$dense_446/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_446/Tensordot/Prod_1?
dense_446/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_446/Tensordot/concat/axis?
dense_446/Tensordot/concatConcatV2!dense_446/Tensordot/free:output:0!dense_446/Tensordot/axes:output:0(dense_446/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_446/Tensordot/concat?
dense_446/Tensordot/stackPack!dense_446/Tensordot/Prod:output:0#dense_446/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_446/Tensordot/stack?
dense_446/Tensordot/transpose	Transposedense_445/Relu:activations:0#dense_446/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_446/Tensordot/transpose?
dense_446/Tensordot/ReshapeReshape!dense_446/Tensordot/transpose:y:0"dense_446/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_446/Tensordot/Reshape?
dense_446/Tensordot/MatMulMatMul$dense_446/Tensordot/Reshape:output:0*dense_446/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_446/Tensordot/MatMul?
dense_446/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_446/Tensordot/Const_2?
!dense_446/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_446/Tensordot/concat_1/axis?
dense_446/Tensordot/concat_1ConcatV2%dense_446/Tensordot/GatherV2:output:0$dense_446/Tensordot/Const_2:output:0*dense_446/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_446/Tensordot/concat_1?
dense_446/TensordotReshape$dense_446/Tensordot/MatMul:product:0%dense_446/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_446/Tensordot?
 dense_446/BiasAdd/ReadVariableOpReadVariableOp)dense_446_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_446/BiasAdd/ReadVariableOp?
dense_446/BiasAddAdddense_446/Tensordot:output:0(dense_446/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_446/BiasAddv
dense_446/ReluReludense_446/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_446/Relu?
"dense_447/Tensordot/ReadVariableOpReadVariableOp+dense_447_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_447/Tensordot/ReadVariableOp~
dense_447/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_447/Tensordot/axes?
dense_447/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_447/Tensordot/free?
dense_447/Tensordot/ShapeShapedense_446/Relu:activations:0*
T0*
_output_shapes
:2
dense_447/Tensordot/Shape?
!dense_447/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_447/Tensordot/GatherV2/axis?
dense_447/Tensordot/GatherV2GatherV2"dense_447/Tensordot/Shape:output:0!dense_447/Tensordot/free:output:0*dense_447/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_447/Tensordot/GatherV2?
#dense_447/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_447/Tensordot/GatherV2_1/axis?
dense_447/Tensordot/GatherV2_1GatherV2"dense_447/Tensordot/Shape:output:0!dense_447/Tensordot/axes:output:0,dense_447/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_447/Tensordot/GatherV2_1?
dense_447/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_447/Tensordot/Const?
dense_447/Tensordot/ProdProd%dense_447/Tensordot/GatherV2:output:0"dense_447/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_447/Tensordot/Prod?
dense_447/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_447/Tensordot/Const_1?
dense_447/Tensordot/Prod_1Prod'dense_447/Tensordot/GatherV2_1:output:0$dense_447/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_447/Tensordot/Prod_1?
dense_447/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_447/Tensordot/concat/axis?
dense_447/Tensordot/concatConcatV2!dense_447/Tensordot/free:output:0!dense_447/Tensordot/axes:output:0(dense_447/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_447/Tensordot/concat?
dense_447/Tensordot/stackPack!dense_447/Tensordot/Prod:output:0#dense_447/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_447/Tensordot/stack?
dense_447/Tensordot/transpose	Transposedense_446/Relu:activations:0#dense_447/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_447/Tensordot/transpose?
dense_447/Tensordot/ReshapeReshape!dense_447/Tensordot/transpose:y:0"dense_447/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_447/Tensordot/Reshape?
dense_447/Tensordot/MatMulMatMul$dense_447/Tensordot/Reshape:output:0*dense_447/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_447/Tensordot/MatMul?
dense_447/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_447/Tensordot/Const_2?
!dense_447/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_447/Tensordot/concat_1/axis?
dense_447/Tensordot/concat_1ConcatV2%dense_447/Tensordot/GatherV2:output:0$dense_447/Tensordot/Const_2:output:0*dense_447/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_447/Tensordot/concat_1?
dense_447/TensordotReshape$dense_447/Tensordot/MatMul:product:0%dense_447/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_447/Tensordot?
 dense_447/BiasAdd/ReadVariableOpReadVariableOp)dense_447_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_447/BiasAdd/ReadVariableOp?
dense_447/BiasAddAdddense_447/Tensordot:output:0(dense_447/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_447/BiasAdd
tf_op_layer_Pow_55/Pow_55/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
tf_op_layer_Pow_55/Pow_55/x?
tf_op_layer_Pow_55/Pow_55Pow$tf_op_layer_Pow_55/Pow_55/x:output:0tf_op_layer_Mul_330/Mul_330:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Pow_55/Pow_55?
tf_op_layer_Mul_331/Mul_331Muldense_447/BiasAdd:z:0tf_op_layer_Pow_55/Pow_55:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_331/Mul_331?
tf_op_layer_Relu_51/Relu_51Relutf_op_layer_Mul_331/Mul_331:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Relu_51/Relu_51?
+tf_op_layer_Max_59/Max_59/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+tf_op_layer_Max_59/Max_59/reduction_indices?
tf_op_layer_Max_59/Max_59Maxinputs_24tf_op_layer_Max_59/Max_59/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
tf_op_layer_Max_59/Max_59?
tf_op_layer_Mul_332/Mul_332Mul)tf_op_layer_Relu_51/Relu_51:activations:0"tf_op_layer_Max_59/Max_59:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_332/Mul_332x
IdentityIdentitytf_op_layer_Mul_332/Mul_332:z:0*
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
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_452242
inputs_0
inputs_1
identitys
Mul_331Mulinputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_331d
IdentityIdentityMul_331:z:0*
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
?
?
E__inference_color_law_layer_call_and_return_conditional_losses_452063

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
?
{
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_452275
inputs_0
inputs_1
identitys
Mul_332Mulinputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_332d
IdentityIdentityMul_332:z:0*
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
?

*__inference_dense_447_layer_call_fn_452225

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
E__inference_dense_447_layer_call_and_return_conditional_losses_4512492
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
h
L__inference_repeat_vector_55_layer_call_and_return_conditional_losses_450902

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
?
k
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_451164

inputs
identity[
	Mul_330/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
	Mul_330/x{
Mul_330MulMul_330/x:output:0inputs*
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
identityIdentity:output:0*+
_input_shapes
:????????? ?:T P
,
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
u
K__inference_concatenate_167_layer_call_and_return_conditional_losses_450985

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
??
?
E__inference_model_111_layer_call_and_return_conditional_losses_451713
inputs_0
inputs_1
inputs_2/
+color_law_tensordot_readvariableop_resource/
+dense_444_tensordot_readvariableop_resource-
)dense_444_biasadd_readvariableop_resource/
+dense_445_tensordot_readvariableop_resource-
)dense_445_biasadd_readvariableop_resource/
+dense_446_tensordot_readvariableop_resource-
)dense_446_biasadd_readvariableop_resource/
+dense_447_tensordot_readvariableop_resource-
)dense_447_biasadd_readvariableop_resource
identity??
repeat_vector_55/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_55/ExpandDims/dim?
repeat_vector_55/ExpandDims
ExpandDimsinputs_0(repeat_vector_55/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
repeat_vector_55/ExpandDims?
repeat_vector_55/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
repeat_vector_55/stack?
repeat_vector_55/TileTile$repeat_vector_55/ExpandDims:output:0repeat_vector_55/stack:output:0*
T0*+
_output_shapes
:????????? 2
repeat_vector_55/Tile?
5tf_op_layer_strided_slice_448/strided_slice_448/beginConst*
_output_shapes
:*
dtype0*
valueB"        27
5tf_op_layer_strided_slice_448/strided_slice_448/begin?
3tf_op_layer_strided_slice_448/strided_slice_448/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_448/strided_slice_448/end?
7tf_op_layer_strided_slice_448/strided_slice_448/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_448/strided_slice_448/strides?
/tf_op_layer_strided_slice_448/strided_slice_448StridedSlicerepeat_vector_55/Tile:output:0>tf_op_layer_strided_slice_448/strided_slice_448/begin:output:0<tf_op_layer_strided_slice_448/strided_slice_448/end:output:0@tf_op_layer_strided_slice_448/strided_slice_448/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_448/strided_slice_448?
5tf_op_layer_strided_slice_451/strided_slice_451/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_451/strided_slice_451/begin?
3tf_op_layer_strided_slice_451/strided_slice_451/endConst*
_output_shapes
:*
dtype0*
valueB"        25
3tf_op_layer_strided_slice_451/strided_slice_451/end?
7tf_op_layer_strided_slice_451/strided_slice_451/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_451/strided_slice_451/strides?
/tf_op_layer_strided_slice_451/strided_slice_451StridedSlicerepeat_vector_55/Tile:output:0>tf_op_layer_strided_slice_451/strided_slice_451/begin:output:0<tf_op_layer_strided_slice_451/strided_slice_451/end:output:0@tf_op_layer_strided_slice_451/strided_slice_451/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask21
/tf_op_layer_strided_slice_451/strided_slice_451?
tf_op_layer_AddV2_110/AddV2_110AddV2inputs_18tf_op_layer_strided_slice_448/strided_slice_448:output:0*
T0*
_cloned(*+
_output_shapes
:????????? 2!
tf_op_layer_AddV2_110/AddV2_110?
5tf_op_layer_strided_slice_450/strided_slice_450/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_450/strided_slice_450/begin?
3tf_op_layer_strided_slice_450/strided_slice_450/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_450/strided_slice_450/end?
7tf_op_layer_strided_slice_450/strided_slice_450/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_450/strided_slice_450/strides?
/tf_op_layer_strided_slice_450/strided_slice_450StridedSlicerepeat_vector_55/Tile:output:0>tf_op_layer_strided_slice_450/strided_slice_450/begin:output:0<tf_op_layer_strided_slice_450/strided_slice_450/end:output:0@tf_op_layer_strided_slice_450/strided_slice_450/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_450/strided_slice_450|
concatenate_167/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_167/concat/axis?
concatenate_167/concatConcatV28tf_op_layer_strided_slice_451/strided_slice_451:output:0#tf_op_layer_AddV2_110/AddV2_110:z:0$concatenate_167/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concatenate_167/concat?
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
color_law/Tensordot/ShapeShape8tf_op_layer_strided_slice_450/strided_slice_450:output:0*
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
color_law/Tensordot/transpose	Transpose8tf_op_layer_strided_slice_450/strided_slice_450:output:0#color_law/Tensordot/concat:output:0*
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
5tf_op_layer_strided_slice_449/strided_slice_449/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_449/strided_slice_449/begin?
3tf_op_layer_strided_slice_449/strided_slice_449/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_449/strided_slice_449/end?
7tf_op_layer_strided_slice_449/strided_slice_449/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_449/strided_slice_449/strides?
/tf_op_layer_strided_slice_449/strided_slice_449StridedSlicerepeat_vector_55/Tile:output:0>tf_op_layer_strided_slice_449/strided_slice_449/begin:output:0<tf_op_layer_strided_slice_449/strided_slice_449/end:output:0@tf_op_layer_strided_slice_449/strided_slice_449/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask21
/tf_op_layer_strided_slice_449/strided_slice_449?
"dense_444/Tensordot/ReadVariableOpReadVariableOp+dense_444_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02$
"dense_444/Tensordot/ReadVariableOp~
dense_444/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_444/Tensordot/axes?
dense_444/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_444/Tensordot/free?
dense_444/Tensordot/ShapeShapeconcatenate_167/concat:output:0*
T0*
_output_shapes
:2
dense_444/Tensordot/Shape?
!dense_444/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_444/Tensordot/GatherV2/axis?
dense_444/Tensordot/GatherV2GatherV2"dense_444/Tensordot/Shape:output:0!dense_444/Tensordot/free:output:0*dense_444/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_444/Tensordot/GatherV2?
#dense_444/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_444/Tensordot/GatherV2_1/axis?
dense_444/Tensordot/GatherV2_1GatherV2"dense_444/Tensordot/Shape:output:0!dense_444/Tensordot/axes:output:0,dense_444/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_444/Tensordot/GatherV2_1?
dense_444/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_444/Tensordot/Const?
dense_444/Tensordot/ProdProd%dense_444/Tensordot/GatherV2:output:0"dense_444/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_444/Tensordot/Prod?
dense_444/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_444/Tensordot/Const_1?
dense_444/Tensordot/Prod_1Prod'dense_444/Tensordot/GatherV2_1:output:0$dense_444/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_444/Tensordot/Prod_1?
dense_444/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_444/Tensordot/concat/axis?
dense_444/Tensordot/concatConcatV2!dense_444/Tensordot/free:output:0!dense_444/Tensordot/axes:output:0(dense_444/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_444/Tensordot/concat?
dense_444/Tensordot/stackPack!dense_444/Tensordot/Prod:output:0#dense_444/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_444/Tensordot/stack?
dense_444/Tensordot/transpose	Transposeconcatenate_167/concat:output:0#dense_444/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_444/Tensordot/transpose?
dense_444/Tensordot/ReshapeReshape!dense_444/Tensordot/transpose:y:0"dense_444/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_444/Tensordot/Reshape?
dense_444/Tensordot/MatMulMatMul$dense_444/Tensordot/Reshape:output:0*dense_444/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_444/Tensordot/MatMul?
dense_444/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_444/Tensordot/Const_2?
!dense_444/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_444/Tensordot/concat_1/axis?
dense_444/Tensordot/concat_1ConcatV2%dense_444/Tensordot/GatherV2:output:0$dense_444/Tensordot/Const_2:output:0*dense_444/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_444/Tensordot/concat_1?
dense_444/TensordotReshape$dense_444/Tensordot/MatMul:product:0%dense_444/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  2
dense_444/Tensordot?
 dense_444/BiasAdd/ReadVariableOpReadVariableOp)dense_444_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_444/BiasAdd/ReadVariableOp?
dense_444/BiasAddAdddense_444/Tensordot:output:0(dense_444/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  2
dense_444/BiasAddu
dense_444/ReluReludense_444/BiasAdd:z:0*
T0*+
_output_shapes
:?????????  2
dense_444/Relu?
tf_op_layer_AddV2_111/AddV2_111AddV2color_law/Tensordot:output:08tf_op_layer_strided_slice_449/strided_slice_449:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2!
tf_op_layer_AddV2_111/AddV2_111?
"dense_445/Tensordot/ReadVariableOpReadVariableOp+dense_445_tensordot_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"dense_445/Tensordot/ReadVariableOp~
dense_445/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_445/Tensordot/axes?
dense_445/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_445/Tensordot/free?
dense_445/Tensordot/ShapeShapedense_444/Relu:activations:0*
T0*
_output_shapes
:2
dense_445/Tensordot/Shape?
!dense_445/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_445/Tensordot/GatherV2/axis?
dense_445/Tensordot/GatherV2GatherV2"dense_445/Tensordot/Shape:output:0!dense_445/Tensordot/free:output:0*dense_445/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_445/Tensordot/GatherV2?
#dense_445/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_445/Tensordot/GatherV2_1/axis?
dense_445/Tensordot/GatherV2_1GatherV2"dense_445/Tensordot/Shape:output:0!dense_445/Tensordot/axes:output:0,dense_445/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_445/Tensordot/GatherV2_1?
dense_445/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_445/Tensordot/Const?
dense_445/Tensordot/ProdProd%dense_445/Tensordot/GatherV2:output:0"dense_445/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_445/Tensordot/Prod?
dense_445/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_445/Tensordot/Const_1?
dense_445/Tensordot/Prod_1Prod'dense_445/Tensordot/GatherV2_1:output:0$dense_445/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_445/Tensordot/Prod_1?
dense_445/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_445/Tensordot/concat/axis?
dense_445/Tensordot/concatConcatV2!dense_445/Tensordot/free:output:0!dense_445/Tensordot/axes:output:0(dense_445/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_445/Tensordot/concat?
dense_445/Tensordot/stackPack!dense_445/Tensordot/Prod:output:0#dense_445/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_445/Tensordot/stack?
dense_445/Tensordot/transpose	Transposedense_444/Relu:activations:0#dense_445/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  2
dense_445/Tensordot/transpose?
dense_445/Tensordot/ReshapeReshape!dense_445/Tensordot/transpose:y:0"dense_445/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_445/Tensordot/Reshape?
dense_445/Tensordot/MatMulMatMul$dense_445/Tensordot/Reshape:output:0*dense_445/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_445/Tensordot/MatMul?
dense_445/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_445/Tensordot/Const_2?
!dense_445/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_445/Tensordot/concat_1/axis?
dense_445/Tensordot/concat_1ConcatV2%dense_445/Tensordot/GatherV2:output:0$dense_445/Tensordot/Const_2:output:0*dense_445/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_445/Tensordot/concat_1?
dense_445/TensordotReshape$dense_445/Tensordot/MatMul:product:0%dense_445/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_445/Tensordot?
 dense_445/BiasAdd/ReadVariableOpReadVariableOp)dense_445_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_445/BiasAdd/ReadVariableOp?
dense_445/BiasAddAdddense_445/Tensordot:output:0(dense_445/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_445/BiasAddv
dense_445/ReluReludense_445/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_445/Relu?
tf_op_layer_Mul_330/Mul_330/xConst*
_output_shapes
: *
dtype0*
valueB
 *??̾2
tf_op_layer_Mul_330/Mul_330/x?
tf_op_layer_Mul_330/Mul_330Mul&tf_op_layer_Mul_330/Mul_330/x:output:0#tf_op_layer_AddV2_111/AddV2_111:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_330/Mul_330?
"dense_446/Tensordot/ReadVariableOpReadVariableOp+dense_446_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_446/Tensordot/ReadVariableOp~
dense_446/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_446/Tensordot/axes?
dense_446/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_446/Tensordot/free?
dense_446/Tensordot/ShapeShapedense_445/Relu:activations:0*
T0*
_output_shapes
:2
dense_446/Tensordot/Shape?
!dense_446/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_446/Tensordot/GatherV2/axis?
dense_446/Tensordot/GatherV2GatherV2"dense_446/Tensordot/Shape:output:0!dense_446/Tensordot/free:output:0*dense_446/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_446/Tensordot/GatherV2?
#dense_446/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_446/Tensordot/GatherV2_1/axis?
dense_446/Tensordot/GatherV2_1GatherV2"dense_446/Tensordot/Shape:output:0!dense_446/Tensordot/axes:output:0,dense_446/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_446/Tensordot/GatherV2_1?
dense_446/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_446/Tensordot/Const?
dense_446/Tensordot/ProdProd%dense_446/Tensordot/GatherV2:output:0"dense_446/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_446/Tensordot/Prod?
dense_446/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_446/Tensordot/Const_1?
dense_446/Tensordot/Prod_1Prod'dense_446/Tensordot/GatherV2_1:output:0$dense_446/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_446/Tensordot/Prod_1?
dense_446/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_446/Tensordot/concat/axis?
dense_446/Tensordot/concatConcatV2!dense_446/Tensordot/free:output:0!dense_446/Tensordot/axes:output:0(dense_446/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_446/Tensordot/concat?
dense_446/Tensordot/stackPack!dense_446/Tensordot/Prod:output:0#dense_446/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_446/Tensordot/stack?
dense_446/Tensordot/transpose	Transposedense_445/Relu:activations:0#dense_446/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_446/Tensordot/transpose?
dense_446/Tensordot/ReshapeReshape!dense_446/Tensordot/transpose:y:0"dense_446/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_446/Tensordot/Reshape?
dense_446/Tensordot/MatMulMatMul$dense_446/Tensordot/Reshape:output:0*dense_446/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_446/Tensordot/MatMul?
dense_446/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_446/Tensordot/Const_2?
!dense_446/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_446/Tensordot/concat_1/axis?
dense_446/Tensordot/concat_1ConcatV2%dense_446/Tensordot/GatherV2:output:0$dense_446/Tensordot/Const_2:output:0*dense_446/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_446/Tensordot/concat_1?
dense_446/TensordotReshape$dense_446/Tensordot/MatMul:product:0%dense_446/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_446/Tensordot?
 dense_446/BiasAdd/ReadVariableOpReadVariableOp)dense_446_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_446/BiasAdd/ReadVariableOp?
dense_446/BiasAddAdddense_446/Tensordot:output:0(dense_446/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_446/BiasAddv
dense_446/ReluReludense_446/BiasAdd:z:0*
T0*,
_output_shapes
:????????? ?2
dense_446/Relu?
"dense_447/Tensordot/ReadVariableOpReadVariableOp+dense_447_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"dense_447/Tensordot/ReadVariableOp~
dense_447/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_447/Tensordot/axes?
dense_447/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_447/Tensordot/free?
dense_447/Tensordot/ShapeShapedense_446/Relu:activations:0*
T0*
_output_shapes
:2
dense_447/Tensordot/Shape?
!dense_447/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_447/Tensordot/GatherV2/axis?
dense_447/Tensordot/GatherV2GatherV2"dense_447/Tensordot/Shape:output:0!dense_447/Tensordot/free:output:0*dense_447/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_447/Tensordot/GatherV2?
#dense_447/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_447/Tensordot/GatherV2_1/axis?
dense_447/Tensordot/GatherV2_1GatherV2"dense_447/Tensordot/Shape:output:0!dense_447/Tensordot/axes:output:0,dense_447/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_447/Tensordot/GatherV2_1?
dense_447/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_447/Tensordot/Const?
dense_447/Tensordot/ProdProd%dense_447/Tensordot/GatherV2:output:0"dense_447/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_447/Tensordot/Prod?
dense_447/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_447/Tensordot/Const_1?
dense_447/Tensordot/Prod_1Prod'dense_447/Tensordot/GatherV2_1:output:0$dense_447/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_447/Tensordot/Prod_1?
dense_447/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_447/Tensordot/concat/axis?
dense_447/Tensordot/concatConcatV2!dense_447/Tensordot/free:output:0!dense_447/Tensordot/axes:output:0(dense_447/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_447/Tensordot/concat?
dense_447/Tensordot/stackPack!dense_447/Tensordot/Prod:output:0#dense_447/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_447/Tensordot/stack?
dense_447/Tensordot/transpose	Transposedense_446/Relu:activations:0#dense_447/Tensordot/concat:output:0*
T0*,
_output_shapes
:????????? ?2
dense_447/Tensordot/transpose?
dense_447/Tensordot/ReshapeReshape!dense_447/Tensordot/transpose:y:0"dense_447/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_447/Tensordot/Reshape?
dense_447/Tensordot/MatMulMatMul$dense_447/Tensordot/Reshape:output:0*dense_447/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_447/Tensordot/MatMul?
dense_447/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_447/Tensordot/Const_2?
!dense_447/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_447/Tensordot/concat_1/axis?
dense_447/Tensordot/concat_1ConcatV2%dense_447/Tensordot/GatherV2:output:0$dense_447/Tensordot/Const_2:output:0*dense_447/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_447/Tensordot/concat_1?
dense_447/TensordotReshape$dense_447/Tensordot/MatMul:product:0%dense_447/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:????????? ?2
dense_447/Tensordot?
 dense_447/BiasAdd/ReadVariableOpReadVariableOp)dense_447_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_447/BiasAdd/ReadVariableOp?
dense_447/BiasAddAdddense_447/Tensordot:output:0(dense_447/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:????????? ?2
dense_447/BiasAdd
tf_op_layer_Pow_55/Pow_55/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
tf_op_layer_Pow_55/Pow_55/x?
tf_op_layer_Pow_55/Pow_55Pow$tf_op_layer_Pow_55/Pow_55/x:output:0tf_op_layer_Mul_330/Mul_330:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Pow_55/Pow_55?
tf_op_layer_Mul_331/Mul_331Muldense_447/BiasAdd:z:0tf_op_layer_Pow_55/Pow_55:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_331/Mul_331?
tf_op_layer_Relu_51/Relu_51Relutf_op_layer_Mul_331/Mul_331:z:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Relu_51/Relu_51?
+tf_op_layer_Max_59/Max_59/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+tf_op_layer_Max_59/Max_59/reduction_indices?
tf_op_layer_Max_59/Max_59Maxinputs_24tf_op_layer_Max_59/Max_59/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:????????? *
	keep_dims(2
tf_op_layer_Max_59/Max_59?
tf_op_layer_Mul_332/Mul_332Mul)tf_op_layer_Relu_51/Relu_51:activations:0"tf_op_layer_Max_59/Max_59:output:0*
T0*
_cloned(*,
_output_shapes
:????????? ?2
tf_op_layer_Mul_332/Mul_332x
IdentityIdentitytf_op_layer_Mul_332/Mul_332:z:0*
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
?
`
4__inference_tf_op_layer_Mul_332_layer_call_fn_452281
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
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_4513272
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
?
p
*__inference_color_law_layer_call_fn_452070

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
E__inference_color_law_layer_call_and_return_conditional_losses_4510212
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
?
y
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_451327

inputs
inputs_1
identityq
Mul_332Mulinputsinputs_1*
T0*
_cloned(*,
_output_shapes
:????????? ?2	
Mul_332d
IdentityIdentityMul_332:z:0*
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
?

*__inference_dense_446_layer_call_fn_452175

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
E__inference_dense_446_layer_call_and_return_conditional_losses_4512032
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
?
?
E__inference_dense_447_layer_call_and_return_conditional_losses_452216

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
M
1__inference_repeat_vector_55_layer_call_fn_450908

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
L__inference_repeat_vector_55_layer_call_and_return_conditional_losses_4509022
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
?

*__inference_dense_445_layer_call_fn_452123

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
E__inference_dense_445_layer_call_and_return_conditional_losses_4511422
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
E__inference_dense_446_layer_call_and_return_conditional_losses_452166

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
P
4__inference_tf_op_layer_Relu_51_layer_call_fn_452258

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
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_4512992
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
?
u
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_450939

inputs
identity?
strided_slice_451/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_451/begin
strided_slice_451/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_451/end?
strided_slice_451/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_451/strides?
strided_slice_451StridedSliceinputs strided_slice_451/begin:output:0strided_slice_451/end:output:0"strided_slice_451/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:????????? *
ellipsis_mask*
end_mask2
strided_slice_451r
IdentityIdentitystrided_slice_451:output:0*
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
?
E__inference_dense_444_layer_call_and_return_conditional_losses_452027

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
?
E__inference_dense_446_layer_call_and_return_conditional_losses_451203

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
O
3__inference_tf_op_layer_Max_59_layer_call_fn_452269

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
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_4513132
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
	input_2247
serving_default_input_224:0????????? ?
G
latent_params6
serving_default_latent_params:0?????????L
tf_op_layer_Mul_3325
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_model??{"class_name": "Model", "name": "model_111", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_111", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}, "name": "latent_params", "inbound_nodes": []}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector_55", "trainable": true, "dtype": "float32", "n": 32}, "name": "repeat_vector_55", "inbound_nodes": [[["latent_params", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}, "name": "conditional_params", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_448", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_448", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_448/begin", "strided_slice_448/end", "strided_slice_448/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_448", "inbound_nodes": [[["repeat_vector_55", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_451", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_451", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_451/begin", "strided_slice_451/end", "strided_slice_451/strides"], "attr": {"end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_451", "inbound_nodes": [[["repeat_vector_55", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_110", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_110", "op": "AddV2", "input": ["conditional_params_58", "strided_slice_448"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_110", "inbound_nodes": [[["conditional_params", 0, 0, {}], ["tf_op_layer_strided_slice_448", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_167", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_167", "inbound_nodes": [[["tf_op_layer_strided_slice_451", 0, 0, {}], ["tf_op_layer_AddV2_110", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_450", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_450", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_450/begin", "strided_slice_450/end", "strided_slice_450/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_450", "inbound_nodes": [[["repeat_vector_55", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_444", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_444", "inbound_nodes": [[["concatenate_167", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "color_law", "inbound_nodes": [[["tf_op_layer_strided_slice_450", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_449", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_449", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_449/begin", "strided_slice_449/end", "strided_slice_449/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_449", "inbound_nodes": [[["repeat_vector_55", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_445", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_445", "inbound_nodes": [[["dense_444", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_111", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_111", "op": "AddV2", "input": ["color_law_58/Identity", "strided_slice_449"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_111", "inbound_nodes": [[["color_law", 0, 0, {}], ["tf_op_layer_strided_slice_449", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_446", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_446", "inbound_nodes": [[["dense_445", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_330", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_330", "op": "Mul", "input": ["Mul_330/x", "AddV2_111"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}, "name": "tf_op_layer_Mul_330", "inbound_nodes": [[["tf_op_layer_AddV2_111", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_447", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_447", "inbound_nodes": [[["dense_446", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow_55", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_55", "op": "Pow", "input": ["Pow_55/x", "Mul_330"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}, "name": "tf_op_layer_Pow_55", "inbound_nodes": [[["tf_op_layer_Mul_330", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_331", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_331", "op": "Mul", "input": ["dense_447/Identity", "Pow_55"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_331", "inbound_nodes": [[["dense_447", 0, 0, {}], ["tf_op_layer_Pow_55", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_224"}, "name": "input_224", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_51", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_51", "op": "Relu", "input": ["Mul_331"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Relu_51", "inbound_nodes": [[["tf_op_layer_Mul_331", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Max_59", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_59", "op": "Max", "input": ["input_224", "Max_59/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Max_59", "inbound_nodes": [[["input_224", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_332", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_332", "op": "Mul", "input": ["Relu_51", "Max_59"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_332", "inbound_nodes": [[["tf_op_layer_Relu_51", 0, 0, {}], ["tf_op_layer_Max_59", 0, 0, {}]]]}], "input_layers": [["latent_params", 0, 0], ["conditional_params", 0, 0], ["input_224", 0, 0]], "output_layers": [["tf_op_layer_Mul_332", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 6]}, {"class_name": "TensorShape", "items": [null, 32, 1]}, {"class_name": "TensorShape", "items": [null, 32, 288]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_111", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}, "name": "latent_params", "inbound_nodes": []}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector_55", "trainable": true, "dtype": "float32", "n": 32}, "name": "repeat_vector_55", "inbound_nodes": [[["latent_params", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}, "name": "conditional_params", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_448", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_448", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_448/begin", "strided_slice_448/end", "strided_slice_448/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_448", "inbound_nodes": [[["repeat_vector_55", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_451", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_451", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_451/begin", "strided_slice_451/end", "strided_slice_451/strides"], "attr": {"end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_451", "inbound_nodes": [[["repeat_vector_55", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_110", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_110", "op": "AddV2", "input": ["conditional_params_58", "strided_slice_448"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_110", "inbound_nodes": [[["conditional_params", 0, 0, {}], ["tf_op_layer_strided_slice_448", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_167", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_167", "inbound_nodes": [[["tf_op_layer_strided_slice_451", 0, 0, {}], ["tf_op_layer_AddV2_110", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_450", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_450", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_450/begin", "strided_slice_450/end", "strided_slice_450/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_450", "inbound_nodes": [[["repeat_vector_55", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_444", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_444", "inbound_nodes": [[["concatenate_167", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "color_law", "inbound_nodes": [[["tf_op_layer_strided_slice_450", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_449", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_449", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_449/begin", "strided_slice_449/end", "strided_slice_449/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_449", "inbound_nodes": [[["repeat_vector_55", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_445", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_445", "inbound_nodes": [[["dense_444", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_111", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_111", "op": "AddV2", "input": ["color_law_58/Identity", "strided_slice_449"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_111", "inbound_nodes": [[["color_law", 0, 0, {}], ["tf_op_layer_strided_slice_449", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_446", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_446", "inbound_nodes": [[["dense_445", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_330", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_330", "op": "Mul", "input": ["Mul_330/x", "AddV2_111"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}, "name": "tf_op_layer_Mul_330", "inbound_nodes": [[["tf_op_layer_AddV2_111", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_447", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_447", "inbound_nodes": [[["dense_446", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow_55", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_55", "op": "Pow", "input": ["Pow_55/x", "Mul_330"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}, "name": "tf_op_layer_Pow_55", "inbound_nodes": [[["tf_op_layer_Mul_330", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_331", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_331", "op": "Mul", "input": ["dense_447/Identity", "Pow_55"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_331", "inbound_nodes": [[["dense_447", 0, 0, {}], ["tf_op_layer_Pow_55", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_224"}, "name": "input_224", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_51", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_51", "op": "Relu", "input": ["Mul_331"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Relu_51", "inbound_nodes": [[["tf_op_layer_Mul_331", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Max_59", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_59", "op": "Max", "input": ["input_224", "Max_59/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Max_59", "inbound_nodes": [[["input_224", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_332", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_332", "op": "Mul", "input": ["Relu_51", "Max_59"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_332", "inbound_nodes": [[["tf_op_layer_Relu_51", 0, 0, {}], ["tf_op_layer_Max_59", 0, 0, {}]]]}], "input_layers": [["latent_params", 0, 0], ["conditional_params", 0, 0], ["input_224", 0, 0]], "output_layers": [["tf_op_layer_Mul_332", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "latent_params", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "RepeatVector", "name": "repeat_vector_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "repeat_vector_55", "trainable": true, "dtype": "float32", "n": 32}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conditional_params", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}}
?
 trainable_variables
!regularization_losses
"	variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_448", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_448", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_448", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_448/begin", "strided_slice_448/end", "strided_slice_448/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "new_axis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}}
?
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_451", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_451", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_451", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_451/begin", "strided_slice_451/end", "strided_slice_451/strides"], "attr": {"end_mask": {"i": "2"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "ellipsis_mask": {"i": "1"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}}
?
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_110", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_110", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_110", "op": "AddV2", "input": ["conditional_params_58", "strided_slice_448"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_167", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_167", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 3]}, {"class_name": "TensorShape", "items": [null, 32, 1]}]}
?
0trainable_variables
1regularization_losses
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_450", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_450", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_450", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_450/begin", "strided_slice_450/end", "strided_slice_450/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}}
?

4kernel
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_444", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_444", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 4]}}
?4

:kernel
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?2
_tf_keras_layer?2{"class_name": "Dense", "name": "color_law", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 1]}}
?
?trainable_variables
@regularization_losses
A	variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_449", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_449", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_449", "op": "StridedSlice", "input": ["repeat_vector_55/Identity", "strided_slice_449/begin", "strided_slice_449/end", "strided_slice_449/strides"], "attr": {"T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}}
?

Ckernel
Dbias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_445", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_445", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32]}}
?
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_111", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_111", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_111", "op": "AddV2", "input": ["color_law_58/Identity", "strided_slice_449"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?

Mkernel
Nbias
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_446", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_446", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 128]}}
?
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_330", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_330", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_330", "op": "Mul", "input": ["Mul_330/x", "AddV2_111"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}}
?

Wkernel
Xbias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_447", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_447", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 256]}}
?
]trainable_variables
^regularization_losses
_	variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Pow_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Pow_55", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_55", "op": "Pow", "input": ["Pow_55/x", "Mul_330"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}}
?
atrainable_variables
bregularization_losses
c	variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_331", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_331", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_331", "op": "Mul", "input": ["dense_447/Identity", "Pow_55"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_224", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_224"}}
?
etrainable_variables
fregularization_losses
g	variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Relu_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Relu_51", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_51", "op": "Relu", "input": ["Mul_331"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
itrainable_variables
jregularization_losses
k	variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Max_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Max_59", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_59", "op": "Max", "input": ["input_224", "Max_59/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": -1}}}
?
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_332", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_332", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_332", "op": "Mul", "input": ["Relu_51", "Max_59"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
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
?
qlayer_regularization_losses
rmetrics
slayer_metrics
trainable_variables
regularization_losses
	variables

tlayers
unon_trainable_variables
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
wmetrics
xlayer_metrics
trainable_variables
regularization_losses
	variables

ylayers
znon_trainable_variables
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
|metrics
}layer_metrics
 trainable_variables
!regularization_losses
"	variables

~layers
non_trainable_variables
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
?metrics
?layer_metrics
$trainable_variables
%regularization_losses
&	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
(trainable_variables
)regularization_losses
*	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
,trainable_variables
-regularization_losses
.	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
0trainable_variables
1regularization_losses
2	variables
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
":  2dense_444/kernel
: 2dense_444/bias
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
 ?layer_regularization_losses
?metrics
?layer_metrics
6trainable_variables
7regularization_losses
8	variables
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$	?2color_law_58/kernel
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?layer_metrics
;trainable_variables
<regularization_losses
=	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
?trainable_variables
@regularization_losses
A	variables
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	 ?2dense_445/kernel
:?2dense_445/bias
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
 ?layer_regularization_losses
?metrics
?layer_metrics
Etrainable_variables
Fregularization_losses
G	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
Itrainable_variables
Jregularization_losses
K	variables
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_446/kernel
:?2dense_446/bias
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
 ?layer_regularization_losses
?metrics
?layer_metrics
Otrainable_variables
Pregularization_losses
Q	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
Strainable_variables
Tregularization_losses
U	variables
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_447/kernel
:?2dense_447/bias
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
 ?layer_regularization_losses
?metrics
?layer_metrics
Ytrainable_variables
Zregularization_losses
[	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
]trainable_variables
^regularization_losses
_	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
atrainable_variables
bregularization_losses
c	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
etrainable_variables
fregularization_losses
g	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
itrainable_variables
jregularization_losses
k	variables
?layers
?non_trainable_variables
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
?metrics
?layer_metrics
mtrainable_variables
nregularization_losses
o	variables
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
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
'
:0"
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
'
:0"
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
!__inference__wrapped_model_450893?
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
	input_224????????? ?
?2?
E__inference_model_111_layer_call_and_return_conditional_losses_451882
E__inference_model_111_layer_call_and_return_conditional_losses_451713
E__inference_model_111_layer_call_and_return_conditional_losses_451380
E__inference_model_111_layer_call_and_return_conditional_losses_451337?
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
*__inference_model_111_layer_call_fn_451932
*__inference_model_111_layer_call_fn_451517
*__inference_model_111_layer_call_fn_451907
*__inference_model_111_layer_call_fn_451449?
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
L__inference_repeat_vector_55_layer_call_and_return_conditional_losses_450902?
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
1__inference_repeat_vector_55_layer_call_fn_450908?
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
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_451940?
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
>__inference_tf_op_layer_strided_slice_448_layer_call_fn_451945?
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
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_451953?
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
>__inference_tf_op_layer_strided_slice_451_layer_call_fn_451958?
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
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_451964?
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
6__inference_tf_op_layer_AddV2_110_layer_call_fn_451970?
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
K__inference_concatenate_167_layer_call_and_return_conditional_losses_451977?
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
0__inference_concatenate_167_layer_call_fn_451983?
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
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_451991?
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
>__inference_tf_op_layer_strided_slice_450_layer_call_fn_451996?
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
E__inference_dense_444_layer_call_and_return_conditional_losses_452027?
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
*__inference_dense_444_layer_call_fn_452036?
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
E__inference_color_law_layer_call_and_return_conditional_losses_452063?
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
*__inference_color_law_layer_call_fn_452070?
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
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_452078?
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
>__inference_tf_op_layer_strided_slice_449_layer_call_fn_452083?
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
E__inference_dense_445_layer_call_and_return_conditional_losses_452114?
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
*__inference_dense_445_layer_call_fn_452123?
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
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_452129?
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
6__inference_tf_op_layer_AddV2_111_layer_call_fn_452135?
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
E__inference_dense_446_layer_call_and_return_conditional_losses_452166?
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
*__inference_dense_446_layer_call_fn_452175?
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
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_452181?
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
4__inference_tf_op_layer_Mul_330_layer_call_fn_452186?
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
E__inference_dense_447_layer_call_and_return_conditional_losses_452216?
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
*__inference_dense_447_layer_call_fn_452225?
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
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_452231?
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
3__inference_tf_op_layer_Pow_55_layer_call_fn_452236?
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
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_452242?
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
4__inference_tf_op_layer_Mul_331_layer_call_fn_452248?
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
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_452253?
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
4__inference_tf_op_layer_Relu_51_layer_call_fn_452258?
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
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_452264?
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
3__inference_tf_op_layer_Max_59_layer_call_fn_452269?
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
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_452275?
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
4__inference_tf_op_layer_Mul_332_layer_call_fn_452281?
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
$__inference_signature_wrapper_451544conditional_params	input_224latent_params?
!__inference__wrapped_model_450893?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_224????????? ?
? "N?K
I
tf_op_layer_Mul_3322?/
tf_op_layer_Mul_332????????? ??
E__inference_color_law_layer_call_and_return_conditional_losses_452063d:3?0
)?&
$?!
inputs????????? 
? "*?'
 ?
0????????? ?
? ?
*__inference_color_law_layer_call_fn_452070W:3?0
)?&
$?!
inputs????????? 
? "?????????? ??
K__inference_concatenate_167_layer_call_and_return_conditional_losses_451977?b?_
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
0__inference_concatenate_167_layer_call_fn_451983?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? "?????????? ?
E__inference_dense_444_layer_call_and_return_conditional_losses_452027d453?0
)?&
$?!
inputs????????? 
? ")?&
?
0?????????  
? ?
*__inference_dense_444_layer_call_fn_452036W453?0
)?&
$?!
inputs????????? 
? "??????????  ?
E__inference_dense_445_layer_call_and_return_conditional_losses_452114eCD3?0
)?&
$?!
inputs?????????  
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_445_layer_call_fn_452123XCD3?0
)?&
$?!
inputs?????????  
? "?????????? ??
E__inference_dense_446_layer_call_and_return_conditional_losses_452166fMN4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_446_layer_call_fn_452175YMN4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
E__inference_dense_447_layer_call_and_return_conditional_losses_452216fWX4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
*__inference_dense_447_layer_call_fn_452225YWX4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
E__inference_model_111_layer_call_and_return_conditional_losses_451337?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_224????????? ?
p

 
? "*?'
 ?
0????????? ?
? ?
E__inference_model_111_layer_call_and_return_conditional_losses_451380?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_224????????? ?
p 

 
? "*?'
 ?
0????????? ?
? ?
E__inference_model_111_layer_call_and_return_conditional_losses_451713?	:45CDMNWX???
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
E__inference_model_111_layer_call_and_return_conditional_losses_451882?	:45CDMNWX???
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
*__inference_model_111_layer_call_fn_451449?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_224????????? ?
p

 
? "?????????? ??
*__inference_model_111_layer_call_fn_451517?	:45CDMNWX???
???
???
'?$
latent_params?????????
0?-
conditional_params????????? 
(?%
	input_224????????? ?
p 

 
? "?????????? ??
*__inference_model_111_layer_call_fn_451907?	:45CDMNWX???
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
*__inference_model_111_layer_call_fn_451932?	:45CDMNWX???
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
L__inference_repeat_vector_55_layer_call_and_return_conditional_losses_450902n8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0????????? ?????????
? ?
1__inference_repeat_vector_55_layer_call_fn_450908a8?5
.?+
)?&
inputs??????????????????
? "%?"????????? ??????????
$__inference_signature_wrapper_451544?	:45CDMNWX???
? 
???
F
conditional_params0?-
conditional_params????????? 
5
	input_224(?%
	input_224????????? ?
8
latent_params'?$
latent_params?????????"N?K
I
tf_op_layer_Mul_3322?/
tf_op_layer_Mul_332????????? ??
Q__inference_tf_op_layer_AddV2_110_layer_call_and_return_conditional_losses_451964?b?_
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
6__inference_tf_op_layer_AddV2_110_layer_call_fn_451970?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? "?????????? ?
Q__inference_tf_op_layer_AddV2_111_layer_call_and_return_conditional_losses_452129?c?`
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
6__inference_tf_op_layer_AddV2_111_layer_call_fn_452135?c?`
Y?V
T?Q
'?$
inputs/0????????? ?
&?#
inputs/1????????? 
? "?????????? ??
N__inference_tf_op_layer_Max_59_layer_call_and_return_conditional_losses_452264a4?1
*?'
%?"
inputs????????? ?
? ")?&
?
0????????? 
? ?
3__inference_tf_op_layer_Max_59_layer_call_fn_452269T4?1
*?'
%?"
inputs????????? ?
? "?????????? ?
O__inference_tf_op_layer_Mul_330_layer_call_and_return_conditional_losses_452181b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
4__inference_tf_op_layer_Mul_330_layer_call_fn_452186U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
O__inference_tf_op_layer_Mul_331_layer_call_and_return_conditional_losses_452242?d?a
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
4__inference_tf_op_layer_Mul_331_layer_call_fn_452248?d?a
Z?W
U?R
'?$
inputs/0????????? ?
'?$
inputs/1????????? ?
? "?????????? ??
O__inference_tf_op_layer_Mul_332_layer_call_and_return_conditional_losses_452275?c?`
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
4__inference_tf_op_layer_Mul_332_layer_call_fn_452281?c?`
Y?V
T?Q
'?$
inputs/0????????? ?
&?#
inputs/1????????? 
? "?????????? ??
N__inference_tf_op_layer_Pow_55_layer_call_and_return_conditional_losses_452231b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
3__inference_tf_op_layer_Pow_55_layer_call_fn_452236U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
O__inference_tf_op_layer_Relu_51_layer_call_and_return_conditional_losses_452253b4?1
*?'
%?"
inputs????????? ?
? "*?'
 ?
0????????? ?
? ?
4__inference_tf_op_layer_Relu_51_layer_call_fn_452258U4?1
*?'
%?"
inputs????????? ?
? "?????????? ??
Y__inference_tf_op_layer_strided_slice_448_layer_call_and_return_conditional_losses_451940`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_448_layer_call_fn_451945S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_449_layer_call_and_return_conditional_losses_452078`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_449_layer_call_fn_452083S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_450_layer_call_and_return_conditional_losses_451991`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_450_layer_call_fn_451996S3?0
)?&
$?!
inputs????????? 
? "?????????? ?
Y__inference_tf_op_layer_strided_slice_451_layer_call_and_return_conditional_losses_451953`3?0
)?&
$?!
inputs????????? 
? ")?&
?
0????????? 
? ?
>__inference_tf_op_layer_strided_slice_451_layer_call_fn_451958S3?0
)?&
$?!
inputs????????? 
? "?????????? 