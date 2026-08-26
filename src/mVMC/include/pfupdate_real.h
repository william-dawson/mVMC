#ifndef _PFUPDATE_REAL
#define _PFUPDATE_REAL
void CalculateNewPfM_real(const int mi, const int s, double *pfMNew_real, const int *eleIdx,
                     const int qpStart, const int qpEnd);
void CalculateNewPfM2_real(const int mi, const int s, double *pfMNew_real, const int *eleIdx,
                     const int qpStart, const int qpEnd);
void UpdateMAll_real(const int mi, const int s, const int *eleIdx,
                const int qpStart, const int qpEnd);
#ifdef USE_GPU_PFAFFIAN
void UpdateMAll_real_gpu(int ma, int s, const int *eleIdx, int qpStart, int qpEnd);
#endif

void CalculateNewPfMBF_real(const int *icount, const int *msaTmp,
                            double *pfMNew, const int *eleIdx,
                            const int qpStart, const int qpEnd, const double *bufM);

void UpdateMAll_BF_real(const int *icount, const int *msaTmp,
                  double *pfMNew, const int *eleIdx,
                  const int qpStart, const int qpEnd) ;

#endif


