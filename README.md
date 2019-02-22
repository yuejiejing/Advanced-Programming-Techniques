# Advanced-Programming-Techniques
Two-Dimensional Discrete Fourier Transform on CPU and GPU

For this assignment you will be implementing THREE different versions of the 2-dimensional DFT. One will be
using CPU threads, one will be using MPI and one will be running on GPUs using CUDA. As well as being graded
for producing the correct output, each submission will be timed and a portion of your grade will be dependent on
how your implementation performs against other implementations in the class. Timing of your code is best done by
submitting batch PBS jobs.
| The first will be a multi-threaded CPU version that utilizes the summation method shown in equation 1.1.
You may use as many threads as you wish but your implementation will be run on a machine with 8 cores.
`nodes=1 ;ppn=8` You MUST use C++'s built in thread capabilities. i.e. don't use the pthreads library directly.
You may synchronize your threads however you wish. Some ways may be more efficient that others. If you want
to find the best approach, you'll need to research and experiment.
| The second implementation will be an MPI implementation of the Danielson-Lanczos Algorithm described
previously in this document. All of your MPI runs should be done using PBS scripts. Please do NOT run
MPI jobs directly on the login node. If you encounter any issues running MPI jobs, immediately submit a help
request to the pace support team. I have been told all previous issues should be fixed. You should be using 8
MPI ranks when testing the performance of your code.
| For the third implementation, you must implement either of the two algorithms to run on GPUs using CUDA.
You may implement either approach however recall your grade will partly be determined by the performance
of the implementation you submit for grading. You may use all of the available resources on the p100 gpus
available on the coc-ice job queue for this implementation.
