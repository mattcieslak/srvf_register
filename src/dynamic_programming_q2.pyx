"# distutils: language=c"
from cython.parallel import parallel, prange
import cython 
import numpy as np
cimport numpy as np
from libc.stdlib cimport abort, malloc, free


# From dpgrid.c
"""
/**
 * 1-D table lookup for a sorted array of query points.
 *
 * Given a partition p and an increasing sequence of numbers tv between 
 * p[0] and p[np-1], computes the sequence of indexes idxv such that for 
 * i=0,...,ntv-1, p[idxv[i]] <= tv[i] < p[idxv[i]+1].  If tv[i]==p[np-1], 
 * then idxv[i] will be set to np-2.
 *
 * param p an increasing sequence (the table)
 * param np the length of \a p
 * param tv an increasing sequence (the query points)
 * param ntv the length of \a tv
 * param idxv [output] pre-allocated array of \a ntv ints to hold result
 */
"""
cdef extern void dp_all_indexes( double *p, int np_, double *tv, int ntv, int *idxv ) nogil


"""
/**
* Computes cost of best path from (0,0) to all other gridpoints.
*
* param Q1 values of the first SRVF
* param T1 changepoint parameters of the first SRVF
* param nsamps1 the length of T1
* param Q2 values of the second SRVF
* param T2 changepoint parameters of the second SRVF
* param nsamps2 the length of T2
* param dim dimension of the ambient space
* param tv1 the Q1 (column) parameter values for the DP grid
* param idxv1 Q1 indexes for tv1, as computed by \c dp_all_indexes()
* param ntv1 the length of tv1
* param tv2 the Q2 (row) parameter values for the DP grid
* param idxv2 Q2 indexes for tv2, as computed by \c dp_all_indexes()
* param ntv2 the length of tv2
* param E [output] on return, E[ntv2*i+j] holds the cost of the best 
*       path from (0,0) to (tv1[i],tv2[j]) in the grid.
* param P [output] on return, P[ntv2*i+j] holds the predecessor of 
*       (tv1[i],tv2[j]).  If predecessor is (tv1[k],tv2[l]), then 
*       P[ntv2*i+j] = k*ntv2+l.
* return E[ntv1*ntv2-1], the cost of the best path from (tv1[0],tv2[0]) 
*         to (tv1[ntv1-1],tv2[ntv2-1]).
*/
"""
cdef extern double dp_costs(
                double *Q1, double *T1, int nsamps1,
                double *Q2, double *T2, int nsamps2,
                int dim, double *tv1, int *idxv1, int ntv1, 
                double *tv2, int *idxv2, int ntv2, 
                double *E, int *P, double lam ) nogil

"""
/**
 * Given predecessor table P, builds the piecewise-linear reparametrization 
 * function gamma.
 *
 * G and T must already be allocated with size max(ntv1,ntv2).  The actual 
 * number of points on gamma will be the return value.
 *
 * param P P[ntv2*i+j] holds the predecessor of (tv1[i],tv2[j]).  If 
 *       predecessor is (tv1[k],tv2[l]), then P[ntv2*i+j] = k*ntv2+l.
 * param tv1 the Q1 (column) parameter values for the DP grid
 * param ntv1 the length of tv1
 * param tv2 the Q2 (row) parameter values for the DP grid
 * param ntv2 the length of tv2
 * param G [output] reparametrization function values
 * param T [output] reparametrization changepoint parameters
 * return the length of G (same as length of T).
 */
"""
cdef extern int dp_build_gamma( 
                int *P, 
                double *tv1, int ntv1, 
                double *tv2, int ntv2,
                double *G, double *T ) nogil


def dp( np.ndarray[double,ndim=1] q1_data, 
        np.ndarray[double,ndim=1] q1_time, 
        np.ndarray[double,ndim=1] q2_data,
        np.ndarray[double,ndim=1] q2_time, 
        np.ndarray[double,ndim=1] input_tv1, 
        np.ndarray[double,ndim=1] input_tv2,
        double lam = 0.0 ):
    cdef double *Q1 = &q1_data[0]
    cdef double *T1 = &q1_time[0]
    cdef double *Q2 = &q2_data[0]
    cdef double *T2 = &q2_time[0]
    cdef double *tv1 = &input_tv1[0]
    cdef double *tv2 = &input_tv2[0]

    cdef int nsamps1, nsamps2, ntv1, ntv2
    cdef int Gsize
    cdef int dim = 1
    cdef double m, rootm;
    cdef int sr, sc #; /* source row and column index */
    cdef int tr, tc #; /* target row and column index */
    cdef int Galloc_size
    cdef double pres

    nsamps1, nsamps2 = q1_time.shape[0], q2_time.shape[0]
    cdef np.ndarray[int,ndim=1] idxv1 = np.zeros(nsamps1,dtype=np.int32)
    cdef np.ndarray[int,ndim=1] idxv2 = np.zeros(nsamps2,dtype=np.int32)

    ntv1, ntv2 = input_tv1.shape[0], input_tv2.shape[0]
    
    #Galloc_size = ntv1>ntv2 ? ntv1 : ntv2;
    if ntv1 > ntv2:
        Galloc_size = ntv1
    else:
        Galloc_size = ntv2

    #/* dp_costs() needs indexes for gridpoints precomputed */
    dp_all_indexes( &q1_time[0], nsamps1, &input_tv1[0], ntv1, &idxv1[0] )
    dp_all_indexes( &q2_time[0], nsamps2, &input_tv2[0], ntv2, &idxv2[0] )

    #/* Compute cost of best path from (0,0) to every other grid point */
    cdef np.ndarray[int, ndim=1] P  = np.zeros(ntv1*ntv2, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] E  = np.zeros(ntv1*ntv2, dtype=np.float)
        # /* P[ntv1*j+i] = predecessor of (tv1[i],tv2[j]) along best path */
    
    pres = dp_costs( &q1_data[0], &q1_time[0], nsamps1, 
                     &q2_data[0], &q2_time[0], nsamps2,
                     dim, 
                     &input_tv1[0], &idxv1[0], ntv1, 
                     &input_tv2[0], &idxv2[0], ntv2, 
                     &E[0], &P[0], lam )

    cdef np.ndarray[double, ndim=1] G  = np.zeros(ntv1*ntv2,dtype=np.float)
    cdef np.ndarray[double, ndim=1] T  = np.zeros(ntv1*ntv2,dtype=np.float)
    #/* Reconstruct best path from (0,0) to (1,1) */
    Gsize = dp_build_gamma( &P[0], &input_tv1[0], ntv1, 
                            &input_tv2[0], ntv2, 
                            &G[0], &T[0] )
    return G[:Gsize].copy(), T[:Gsize].copy()
    
cdef void dp_omp( double *Q1, double *T1, 
                  double *Q2, double *T2, 
                  double *tv1, double *tv2,
                  double *output_row,
                  double lam, 
                  int nsamps1, int nsamps2) nogil:
    
    """ 
    cdef double *Q1 = &q1_data[0]
    cdef double *T1 = &q1_time[0]
    cdef double *Q2 = &q2_data[0]
    cdef double *T2 = &q2_time[0]
    cdef double *tv1 = &input_tv1[0]
    cdef double *tv2 = &input_tv2[0]
    """
    
    cdef int ntv1, ntv2
    cdef int Gsize
    cdef int dim = 1
    cdef double m, rootm;
    cdef int sr, sc #; /* source row and column index */
    cdef int tr, tc #; /* target row and column index */
    cdef int Galloc_size
    cdef double pres
    ntv1 = nsamps1
    ntv2 = nsamps2

    # Allocate thread-safe 
    cdef int *idxv1
    cdef int *idxv2
    idxv1 = <int *>malloc(sizeof(int) * nsamps1)
    idxv2 = <int *>malloc(sizeof(int) * nsamps2)
    if idxv1 == NULL or idxv2 == NULL:
        abort()
    
    #Galloc_size = ntv1>ntv2 ? ntv1 : ntv2;
    if ntv1 > ntv2:
        Galloc_size = ntv1
    else:
        Galloc_size = ntv2

    #/* dp_costs() needs indexes for gridpoints precomputed */
    dp_all_indexes( T1, nsamps1, tv1, ntv1, idxv1 )
    dp_all_indexes( T2, nsamps2, tv2, ntv2, idxv2 )

    #/* Compute cost of best path from (0,0) to every other grid point */
    cdef int *P
    cdef double *E
    P = <int *>malloc(sizeof(int) * ntv1 * ntv2)
    E = <double *>malloc(sizeof(double) * ntv1 * ntv2)
    if P == NULL or E == NULL:
        abort()
        # /* P[ntv1*j+i] = predecessor of (tv1[i],tv2[j]) along best path */
    
    pres = dp_costs( Q1, T1, nsamps1, 
                     Q2, T2, nsamps2,
                     dim, 
                     tv1, idxv1, ntv1, 
                     tv2, idxv2, ntv2, 
                     E, P, lam )

    cdef double *G
    cdef double *T
    G = <double *>malloc(sizeof(double) * ntv1 *ntv2)
    T = <double *>malloc(sizeof(double) * ntv1 *ntv2)
    
    #/* Reconstruct best path from (0,0) to (1,1) */
    Gsize = dp_build_gamma( P, tv1, ntv1, 
                            tv2, ntv2, 
                            G, T )
    free(idxv1)
    free(idxv2)
    free(E)
    free(P)
    
    approx(Gsize, T, G, nsamps1, T1, output_row)
    
    free(G)
    free(T)
    
    rescale_gamma(nsamps1, output_row)

@cython.boundscheck(False)
cdef void approx(int nd, double *xd, double *yd, int ni, double *xi, double *yi) nogil:
  cdef int i,k
  cdef double t
  for i in range(ni):
      if xi[i] <= xd[0]:
          t = ( xi[i] - xd[0] ) / ( xd[1] - xd[0] )
          yi[i] = ( 1.0 - t ) * yd[0] + t * yd[1]
      elif  xd[nd-1] <= xi[i]:
          t = ( xi[i] - xd[nd-2] ) / ( xd[nd-1] - xd[nd-2] )
          yi[i] = ( 1.0 - t ) * yd[nd-2] + t * yd[nd-1]
      else:
          for k in range(1, nd):
              if xd[k-1] <= xi[i] and xi[i] <= xd[k]:
                  t = ( xi[i] - xd[k-1] ) / ( xd[k] - xd[k-1] )
                  yi[i] = ( 1.0 - t ) * yd[k-1] + t * yd[k]
                  break

def custom_approx(interp_times, data_times, data_values):
    cdef int nd, ni 
    nd = interp_times.shape[0]
    ni = data_times.shape[0]
    
@cython.boundscheck(False)
cdef void rescale_gamma(int ngamma, double *gamma_vec) nogil:
    cdef int gamma_index
    cdef double denominator = gamma_vec[ngamma-1] - gamma_vec[0]
    for gamma_index in range(ngamma):
        gamma_vec[gamma_index] = (gamma_vec[gamma_index] - gamma_vec[0]) / denominator

@cython.boundscheck(False)
def parallel_dp( np.ndarray[double,ndim=1] q1_data, 
        np.ndarray[double,ndim=1] q1_time, 
        np.ndarray[double,ndim=2] q2_matrix,
        np.ndarray[double,ndim=1] q2_time, 
        np.ndarray[double,ndim=1] input_tv1, 
        np.ndarray[double,ndim=1] input_tv2,
        double lam = 0.0 ):
    
    # These get filled in over each iteration
    cdef int nsamps1, nsamps2
    cdef size_t num_inputs = q2_matrix.shape[0]
    nsamps1, nsamps2 = q1_time.shape[0], q2_time.shape[0]
    if not nsamps1 == nsamps2:
        raise ValueError
        
    # Hold the outputs in gam
    gam_matrix = np.zeros((num_inputs,nsamps1), dtype=np.float)
    cdef double[:,:] gam = gam_matrix
    cdef double[:,:] q2_view = q2_matrix
    cdef int input_num
    with nogil, parallel():
        for input_num in prange(num_inputs, schedule="guided"):
            dp_omp(&q1_data[0], &q1_time[0], 
                   &q2_view[input_num,0], &q2_time[0],
                   &input_tv1[0], &input_tv2[0], 
                   &gam[input_num,0], lam,
                   nsamps1, nsamps2)
                    
    return np.asarray(gam)
