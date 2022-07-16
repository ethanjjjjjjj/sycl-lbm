

#include <CL/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>
using namespace cl::sycl;

#define NSPEEDS 9

void usage(const char* exe){
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

float av_velocity(int nx,int ny, float** cells, bool*  obstacles){
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < ny; jj++){
    for (int ii = 0; ii < nx; ii++){
      /* ignore occupied cells */
      if (!obstacles[ii + jj*nx]){
        /* local density total */
        float local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++){
          local_density += cells[kk][ii + jj*nx];
        }
        float u_x = (cells[1][ii + jj*nx]+ cells[5][ii + jj*nx]+ cells[8][ii + jj*nx]- (cells[3][ii + jj*nx]+cells[6][ii + jj*nx]+cells[7][ii + jj*nx]))/ local_density;
        float u_y = (cells[2][ii + jj*nx]+ cells[5][ii + jj*nx]+ cells[6][ii + jj*nx]- (cells[4][ii + jj*nx]+ cells[7][ii + jj*nx]+ cells[8][ii + jj*nx]))/ local_density;
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}


float calc_reynolds(int nx,int ny,float omega,int reynolds_dim, float** cells, bool* obstacles){
  const float viscosity = 1.f / 6.f * (2.f / omega - 1.f);
  return av_velocity(nx,ny, cells, obstacles) * reynolds_dim / viscosity;
}


int main(int argc, char* argv[]){
    queue q; //
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    //init
    if (argc != 3){
    usage(argv[0]);
  }
  else{
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

    //device device =  q.get_device();
    //std::cout<<device.get_info<info::device::local_mem_size>()<<std::endl;


    struct timeval timstr;                                                             /* structure to hold elapsed time */
    double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

    gettimeofday(&timstr, NULL);
    tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    init_tic=tot_tic;
    FILE*   fp;            /* file pointer */

    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */
      /* open the parameter file */

    int    nx;            /* no. of cells in x-direction */
    int    ny;            /* no. of cells in y-direction */
    int    maxIters;      /* no. of iterations */
    int    reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */

    fp = fopen(paramfile, "r");

  /* read in the parameter values */
    fscanf(fp, "%d\n", &(nx));
    fscanf(fp, "%d\n", &(ny));
    fscanf(fp, "%d\n", &(maxIters));
    fscanf(fp, "%d\n", &(reynolds_dim));
    fscanf(fp, "%f\n", &(density));
    fscanf(fp, "%f\n", &(accel));
    fscanf(fp, "%f\n", &(omega));
    fclose(fp);



    //float* cells[NSPEEDS];
    float** cells=malloc_shared<float*>(NSPEEDS,q);
    float** tmp_cells=malloc_shared<float*>(NSPEEDS,q);

    bool* obstacles=malloc_shared<bool>(nx*ny,q);


 

    for(int i=0;i<NSPEEDS;i++){
        cells[i]=malloc_shared<float>(nx*ny,q);
        tmp_cells[i]=malloc_shared<float>(nx*ny,q);
    }

      /* initialise densities */
  const float w0 = density * 4.f / 9.f;
  float w1 = density      / 9.f;
  float w2 = density      / 36.f;

    q.parallel_for(range<2>(ny,nx),[=](id<2>xy) {
        int jj=xy[0];
        int ii=xy[1];
        cells[0][ii+nx*jj]=w0;
        cells[1][ii+nx*jj]=w1;
        cells[2][ii+nx*jj]=w1;
        cells[3][ii+nx*jj]=w1;
        cells[4][ii+nx*jj]=w1;
        cells[5][ii+nx*jj]=w2;
        cells[6][ii+nx*jj]=w2;
        cells[7][ii+nx*jj]=w2;
        cells[8][ii+nx*jj]=w2;
        obstacles[ii + jj*nx] = 0;
    }).wait();

   /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  int    xx, yy;         /* generic array indices */
  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF){
    obstacles[xx + yy*nx] = blocked;
  }

  /* and close the file */
  fclose(fp);


/* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;
  const int ji = ny - 2;

  int* totcells=malloc_shared<int>(nx*ny,q);
  float* totu=malloc_shared<float>(nx*ny,q);
  float* av_vels=malloc_shared<float>(maxIters,q);

    /* compute weighting factors */
  w1 = density * accel / 9.f;
  w2 = density * accel / 36.f;
    
 


    for(int tt=0;tt<maxIters;tt++){

        //printf("%f \n",cells[5][64*64]);
     q.parallel_for(range<1>(nx),[=](id<1> ii){
        if (!obstacles[ii + ji*nx]
            && (cells[3][ii + ji*nx] - w1) > 0.f
            && (cells[6][ii + ji*nx] - w2) > 0.f
            && (cells[7][ii + ji*nx] - w2) > 0.f){
      /* increase 'east-side' densities */
        cells[1][ii + ji*nx] += w1;
        cells[5][ii + ji*nx] += w2;
        cells[8][ii + ji*nx] += w2;
        /* decrease 'west-side' densities */
        cells[3][ii + ji*nx] -= w1;
        cells[6][ii + ji*nx] -= w2;
        cells[7][ii + ji*nx] -= w2;
         }
    }
    
    ).wait();

    int tot_cells=0;
    float tot_u=0;
    
    q.parallel_for<>(range<2>(ny,nx),[=](id<2> xy){
    int ii=xy.get(1);
    int jj=xy.get(0);

    int y_s = (jj+ny-1)%ny;
    int y_n = (jj + 1)%ny;

    int x_e=(ii+1)%nx;
    int x_w=(ii+nx-1)%nx;

    float rearrange[NSPEEDS];

    rearrange[0] = cells[0][ii + jj*nx]; /* central cell, no movement */
    rearrange[1] = cells[1][x_w + jj*nx]; /* east */
    rearrange[2] = cells[2][ii + y_s*nx]; /* north */
    rearrange[3] = cells[3][x_e + jj*nx]; /* west */
    rearrange[4] = cells[4][ii + y_n*nx]; /* south */
    rearrange[5] = cells[5][x_w + y_s*nx]; /* north-east */
    rearrange[6] = cells[6][x_e + y_s*nx]; /* cnorth-west */
    rearrange[7] = cells[7][x_e + y_n*nx]; /* south-west */
    rearrange[8] = cells[8][x_w + y_n*nx]; /* south-east */

    if (obstacles[jj*nx + ii]){
      tmp_cells[0][ii + jj*nx] = rearrange[0];
      tmp_cells[1][ii + jj*nx] = rearrange[3];
      tmp_cells[2][ii + jj*nx] = rearrange[4];
      tmp_cells[3][ii + jj*nx] = rearrange[1];
      tmp_cells[4][ii + jj*nx] = rearrange[2];
      tmp_cells[5][ii + jj*nx] = rearrange[7];
      tmp_cells[6][ii + jj*nx] = rearrange[8];
      tmp_cells[7][ii + jj*nx] = rearrange[5];
      tmp_cells[8][ii + jj*nx] = rearrange[6];
    }

    else{
      const float c_sq = 1.f / 3.f; /* square of speed of sound */
      const float w0 = 4.f / 9.f;  /* weighting factor */
      const float w1 = 1.f / 9.f;  /* weighting factor */
      const float w2 = 1.f / 36.f; /* weighting factor */
      /* compute local density total */
      float local_density = 0.f;
      for (int kk = 0; kk < NSPEEDS; kk++){
        local_density += rearrange[kk];
      }

      const float u_x = (rearrange[1]+ rearrange[5]+ rearrange[8]- (rearrange[3]+ rearrange[6]+ rearrange[7]))/ local_density;
      const float u_y = (rearrange[2]+ rearrange[5]+ rearrange[6]- (rearrange[4]+ rearrange[7] + rearrange[8]))/ local_density;
      const float u_sq = u_x * u_x + u_y * u_y;
      
      float u[8];
      u[0] =   u_x;        /* east */
      u[1] =         u_y;  /* north */
      u[2] = - u_x;        /* west */
      u[3] =       - u_y;  /* south */
      u[4] =   u_x + u_y;  /* north-east */
      u[5] = - u_x + u_y;  /* north-west */
      u[6] = - u_x - u_y;  /* south-west */
      u[7] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];

      /* zero velocity density: weight w0 */
      d_equ[0] = w0 * local_density* (1.f - u_sq / (2.f * c_sq));
      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[0] / c_sq+ (u[0] * u[0]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
      d_equ[2] = w1 * local_density * (1.f + u[1] / c_sq+ (u[1] * u[1]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
      d_equ[3] = w1 * local_density * (1.f + u[2] / c_sq+ (u[2] * u[2]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
      d_equ[4] = w1 * local_density * (1.f + u[3] / c_sq+ (u[3] * u[3]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[4] / c_sq+ (u[4] * u[4]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
      d_equ[6] = w2 * local_density * (1.f + u[5] / c_sq+ (u[5] * u[5]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
      d_equ[7] = w2 * local_density * (1.f + u[6] / c_sq+ (u[6] * u[6]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
      d_equ[8] = w2 * local_density * (1.f + u[7] / c_sq+ (u[7] * u[7]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));

      tmp_cells[0][ii + jj*nx] = rearrange[0]+ omega * (d_equ[0] - rearrange[0]);
      tmp_cells[1][ii + jj*nx] = rearrange[1]+ omega * (d_equ[1] - rearrange[1]);
      tmp_cells[2][ii + jj*nx] = rearrange[2]+ omega * (d_equ[2] - rearrange[2]);
      tmp_cells[3][ii + jj*nx] = rearrange[3]+ omega * (d_equ[3] - rearrange[3]);
      tmp_cells[4][ii + jj*nx] = rearrange[4]+ omega * (d_equ[4] - rearrange[4]);
      tmp_cells[5][ii + jj*nx] = rearrange[5]+ omega * (d_equ[5] - rearrange[5]);
      tmp_cells[6][ii + jj*nx] = rearrange[6]+ omega * (d_equ[6] - rearrange[6]);
      tmp_cells[7][ii + jj*nx] = rearrange[7]+ omega * (d_equ[7] - rearrange[7]);
      tmp_cells[8][ii + jj*nx] = rearrange[8]+ omega * (d_equ[8] - rearrange[8]);
        
        
      /* accumulate the norm of x- and y- velocity components */
      totu[ii + jj*nx] = sqrtf(u_sq);
      /* increase counter of inspected cells */
      totcells[ii + jj*nx]=1;
    }

    }).wait();


    /*
    for(int i=0;i<nx*ny;i++){
      tot_u+=totu[i];
      tot_cells+=totcells[i];
    }
    */

    av_vels[tt]=tot_u/(float)tot_cells;
    //printf("avvels %f \n",av_vels[tt]);

    //printf("before cells: %f tmp_cells: %f\n",cells[5][64*64],tmp_cells[5][64*64]);
    float** placeholder=cells;
    cells=tmp_cells;
    tmp_cells=placeholder;
    //printf("after cells: %f tmp_cells: %f\n",cells[5][64*64],tmp_cells[5][64*64]);
    }

/* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;



  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(nx,ny,omega,reynolds_dim, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);

    //---------------------------------

    for(int i=0;i<NSPEEDS;i++){
        free(cells[i],q);
    }
    free(cells,q);
    free(obstacles,q);
    free(av_vels,q);


}

