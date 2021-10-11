#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double correlation (double* x, double* y, int size, double invalid)
{
  /* return the r value of a linear correlation
     see NI Fisher page 145, 6.19 */
  double sum_x=0;
  double sum_y=0;
  double mean_x=0;
  double mean_y=0;
  double sum_x_mean=0;
  double sum_y_mean=0;
  double sum_prod_diff_mean=0;
  double r;
  int n=0;
  for (int i = 0; i < size; i++)
    {
      if (x[i]!=invalid && y[i]!=invalid)
	    {
	      sum_x=sum_x+x[i];
	      sum_y=sum_y+y[i];
	      n++;
	    }
    }
  mean_x=sum_x/n;
  mean_y=sum_y/n;
  for (int i = 0; i < size; i++)
    {
      if (x[i]!=invalid && y[i]!=invalid)
	{
	  sum_x_mean=sum_x_mean+pow((x[i]-mean_x),2);
	  sum_y_mean=sum_y_mean+pow((y[i]-mean_y),2);
	  sum_prod_diff_mean=sum_prod_diff_mean+((x[i]-mean_x)*(y[i]-mean_y));
	}
    }
  if (sum_x_mean == 0 || sum_y_mean == 0)
    {
      r = 0;
    }
  else
    {
      r=sum_prod_diff_mean/sqrt((sum_x_mean*sum_y_mean));
    }
  // allow for some small rounding error, this should be negligable for most analysis
  if(r<-1.0&&r>-1.00000000001)
    r=-1.0;
  if(r>1.0&&r<1.00000000001)
    r=1.0;
  if (r<-1.0||r>1.0) 
    {
      printf("problem with correlation function, value of r out of range: %lf\n",r);
      printf("size: %d n: %d\n",size,n);
    }
  return r;
}

void map_autocorrelation(double *one_place, // pointer to one place field map
			 double *one_auto, // pointer to one spatial autocorrelation map
			 int x_bins_place_map, // x size of the place field map (num bins)
			 int y_bins_place_map, // y size of the place field map
			 int x_bins_auto_map, // x size of the autocorrelation map
			 int y_bins_auto_map, // y size of the autocorreltion map
			 int min_for_correlation) // minimum of valid values to do the correlation
{
/*************************************************************
 funciton to do the spatial autocorrelation for a place firing
map. 
 one_place should have a size = x_bins_place_map*y_bins_place_map
 x_bins_auto_map should = (x_bins_place_map*2)+1
 y_bins_auto_map should = (y_bins_place_map*2)+1
 one_auto should have a size =  x_bins_auto_map * y_bins_auto_map
*************************************************************/
  int min_x_offset = 0 - x_bins_place_map;
  int max_x_offset = x_bins_place_map;
  int min_y_offset = 0 - y_bins_place_map;
  int max_y_offset = y_bins_place_map;
  int mid_x = x_bins_place_map; // mid x value in autocorrelation
  int mid_y = y_bins_place_map; // min y value in autocorrelation
  int auto_x;
  int auto_y;
  int index_1;
  int index_2;
  int index_auto;
  int total_bins_place_map = x_bins_place_map * y_bins_place_map;
  int total_bins_auto_map = x_bins_auto_map * y_bins_auto_map;
  int offset_x;
  int offset_y;
 
  int n;
  double r;

  double* value_1_correlation;
  double* value_2_correlation;
  value_1_correlation = (double*)malloc(total_bins_place_map*sizeof(double));
  value_2_correlation = (double*)malloc(total_bins_place_map*sizeof(double));
  // set the auto_place map to -2, this is the invalid value, correlation range from -1 to 1
  for (int i = 0; i < total_bins_auto_map ; i++)
    {
      one_auto[i] = -2;
    }
  // loop for all possible lags in the x axis
  for (int x_off = min_x_offset; x_off <= max_x_offset; x_off++)
    {
      // loop for all possible lags in the y axis
      for (int y_off = min_y_offset; y_off <= max_y_offset; y_off++ )
	{
	  // for all the possible lags, calculate the following values
	  n = 0;  // number of valid lags to do the correlation for this offset
	  r = 0;  // r value of the correlation
	  // loop for all bins in the place map
	  for(int x = 0; x < x_bins_place_map; x++)
	    {
	      for(int y = 0; y < y_bins_place_map; y++)
		{
		  offset_x = x+x_off;
		  offset_y = y+y_off;
		  if ((offset_x >=0 && offset_x < x_bins_place_map) && (offset_y >= 0 && offset_y < y_bins_place_map))
		    {
		      index_1 = (x*y_bins_place_map) + y; // that is the index for the current bin in the place firing rate map
		      index_2 = ((offset_x)*y_bins_place_map)+(offset_y); // that is the index in the offset bin relative to the current bin
		      if (one_place[index_1]!=-1.0 &&one_place[index_2]!=-1.0) // -1 is the invalid value in the place firing rate map, only take into account the data if not invalid value 
			{
			  // copy the value into 2 vectors for the correlation
			  value_1_correlation[n]=one_place[index_1];
			  value_2_correlation[n]=one_place[index_2];
			  n++; 
			}
		    }
		}   
	    }
	  // if enough valid data to calculate the r value, if not the value for this lag will stay -2 
	  if ( n > min_for_correlation)
	    {
	      // calculate a correlation
	      r = correlation(value_1_correlation,value_2_correlation,n,-1.0);
	      auto_x = mid_x + x_off;
	      auto_y = mid_y + y_off;
	      index_auto= (auto_x*y_bins_auto_map)+auto_y;
	      one_auto[index_auto]=r;
	    }
	}
    }
  free(value_1_correlation);
  free(value_2_correlation);
}


