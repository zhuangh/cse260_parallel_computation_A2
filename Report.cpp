// 
// Performs various reporting functions
//
// Do not change the code in this file, as doing so
// could cause your submission to be graded incorrectly
//
////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include "types.h"

using namespace std;

double gflops(int n, int reps, double time){

    // Total number of entries
    long long int n2 = n;
    n2 *= n;
    // Updates
    const long long int updates =  n2 * (long long) reps;
    // Number of flops
    const long long int flops =  (long long ) n * 2L * updates;
    double flop_rate = (double) flops / time;
    return ( flop_rate/1.0e9);
}

void printTOD(string mesg)
{
        time_t tim = time(NULL);
        string s = ctime(&tim);
        if (mesg.length() ==  0) 
            cout << "Time of day: " << s.substr(0,s.length()-1) << endl;
        else {
            cout << "[" << mesg << "] " ;
            cout << s.substr(0,s.length()-1) << endl;
        }
        cout << endl;
}

void perfString(int n, int ntx, int nty, int reps, double t_h, double gflops_h, double t_d, double gflops_d)
{
    printf("\n      N     TX     TY    Reps       t_h     GF_h      t_d      GF_d\n");
    printf("@ %6d  %4d   %4d    %4d   %8.2e  %6.1f  %8.2e  %6.1f\n\n",n,ntx, nty,reps,t_h,gflops_h,t_d,gflops_d);
}
