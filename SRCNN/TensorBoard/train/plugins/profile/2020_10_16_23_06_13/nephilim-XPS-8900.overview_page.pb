�	�J�_V@�J�_V@!�J�_V@	\��`��?\��`��?!\��`��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�J�_V@��M(D�?Ae���PV@YZ�xZ~�?*	��v���`@2F
Iterator::Model�d�VA�?!���Z�G@)��˚X�?1�+C��@=@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate&9{ڡ?!d	m�F:@)׉��
D�?1�}7@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat��	K<��?!�Z&2�5@)lC�8�?1��#[03@:Preprocessing2S
Iterator::Model::ParallelMap�%Tp�?!8���1@)�%Tp�?18���1@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip"����?!�a/�aJ@):���u?1㽟�@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceס����q?!*s�%
@)ס����q?1*s�%
@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor1%��ed?!C�%�a�?)1%��ed?1C�%�a�?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap6��g΢?!|����;@)��҈�}^?1yq�l�o�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��M(D�?��M(D�?!��M(D�?      ��!       "      ��!       *      ��!       2	e���PV@e���PV@!e���PV@:      ��!       B      ��!       J	Z�xZ~�?Z�xZ~�?!Z�xZ~�?R      ��!       Z	Z�xZ~�?Z�xZ~�?!Z�xZ~�?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 