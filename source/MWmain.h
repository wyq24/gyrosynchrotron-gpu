#pragma once

int FindLocalJK(double *nu, int *Lparms, double *Rparms, double *Parms,
	            double *E_arr, double *mu_arr, double *f_arr,
	            double *jX, double *jO, double *kX, double *kO, double *ne_total);
int MW_Transfer(int *Lparms, double *Rparms, double *Parms, double *E_arr, double *mu_arr, double *f_arr, double *RL);
