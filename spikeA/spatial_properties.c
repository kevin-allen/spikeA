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


void map_crosscorrelation(double *one_place, // pointer to a first firing rate map
                          double *two_place, // pointer to a second firing rate map
			 double *one_cross, // pointer to one spatial autocorrelation map
			 int x_bins_place_map, // x size of the place field map (num bins)
			 int y_bins_place_map, // y size of the place field map
			 int x_bins_auto_map, // x size of the autocorrelation map
			 int y_bins_auto_map, // y size of the autocorreltion map
			 int min_for_correlation) // minimum of valid values to do the correlation
{
/*************************************************************
 funciton to do the spatial crosscorrelation between 2 firing rate maps
 
 The invalid values are expected to be set at -1.0
 
 one_place should have a size = x_bins_place_map*y_bins_place_map
 two_place should have a size = x_bins_place_map*y_bins_place_map
 
 x_bins_auto_map should = (x_bins_place_map*2)+1
 y_bins_auto_map should = (y_bins_place_map*2)+1
 one_cross should have a size =  x_bins_auto_map * y_bins_auto_map
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
      one_cross[i] = -2;
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
		      if (one_place[index_1]!=-1.0 &&two_place[index_2]!=-1.0) // -1 is the invalid value in the place firing rate map, only take into account the data if not invalid value 
			{
			  // copy the value into 2 vectors for the correlation
			  value_1_correlation[n]=one_place[index_1];
			  value_2_correlation[n]=two_place[index_2];
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
	      one_cross[index_auto]=r;
	    }
	}
    }
  free(value_1_correlation);
  free(value_2_correlation);
}





void remove_internal_bins_from_border(int num_bins_x, int num_bins_y, int* border_map, int* border_x, int* border_y, int* num_bins_border)
{
  // takes care of the "border" bins that are in the middle of the map
  
  // is_border needs to be above 0 for a border pixel to be kept
  int min;
  int max;
  int min_index;
  int max_index;
  int* is_border= (int*) malloc(sizeof(int)* (*num_bins_border));
  for(int i = 0; i < (*num_bins_border); i++)
    is_border[i]=0;

  // find the outside bins for each row
  for(int x = 0; x < num_bins_x; x++)
  {
    min=num_bins_x;
    max=0;
    min_index=-1;
    max_index=-1;
    for(int i = 0; i < (*num_bins_border);i++)
    {
      if(border_x[i]==x)
      {
        if(border_y[i]>max){
          max=border_y[i];
          max_index=i;
        }
        if(border_y[i]<min){
          min=border_y[i];
          min_index=i;
        }
      }
    }
    // for this row, keep only the extrem values
    if(max_index!=-1)
      is_border[max_index]=1;
    if(min_index!=-1)
      is_border[min_index]=1;
  }
  // find the outside bins for each column
  for(int y = 0; y < num_bins_y; y++)
  {
    min=num_bins_y;
    max=0;
    min_index=-1;
    max_index=-1;
    for(int i = 0; i < (*num_bins_border);i++)
    {
      if(border_y[i]==y)
      {
        if(border_x[i]>max){
          max=border_x[i];
          max_index=i;
        }
        if(border_x[i]<min){
          min=border_x[i];
          min_index=i;
        }
      }
    }
     // for this row, keep only the extrem values
     if(max_index!=-1)
       is_border[max_index]=1;
     if(min_index!=-1)
       is_border[min_index]=1;
    }

    int new_num_bins=0;
  for(int i = 0; i < (*num_bins_border);i++)
  {
    if(is_border[i]>0)
      new_num_bins++;
  }

  
  // remove the bins that is_border==0
  int removed=0;
  for(int i = 0; i < (*num_bins_border);i++)
  {
    if(is_border[i]==0)
    {
      // remove from the map
      border_map[(border_x[i]*num_bins_y)+border_y[i]]=0;
      removed++;
    }
    else
    {
      border_x[i-removed]=border_x[i];
      border_y[i-removed]=border_y[i];
    }
  }
  *num_bins_border=new_num_bins;
  free(is_border);
}

int find_border_starting_point(double* occ_map, int num_bins_x, int num_bins_y,
                               int* border_map,
                               int* border_x,int* border_y,int* num_bins_border)
{
  //printf("find_border_starting_point\n");
  //printf("num_bins_x: %d, num_bins_y: %d\n",num_bins_x,num_bins_y);
  for (int x = 0 ; x < num_bins_x; x++)
  {
    for (int y = 0; y < num_bins_y; y++)
    {
      if (x!=0&&x!=num_bins_x-1&&y!=0&&y!=num_bins_y-1)
      {  
        // a starting point could be a valid pixel, with 3 non visited pixels on right, and none on the left, or reverse
        if((border_map[((x)*num_bins_y)+y]!=1 &&
           occ_map[((x)*num_bins_y)+y]!=-1 &&
           occ_map[((x-1)*num_bins_y)+y-1]==-1 &&
           occ_map[((x-1)*num_bins_y)+y]==-1 &&
           occ_map[((x-1)*num_bins_y)+y+1]==-1 &&
           occ_map[((x+1)*num_bins_y)+y-1]!=-1 &&
           occ_map[((x+1)*num_bins_y)+y]!=-1 && 
           occ_map[((x+1)*num_bins_y)+y+1]!=-1)
             ||
               (border_map[((x)*num_bins_y)+y]!=1 &&
               occ_map[((x)*num_bins_y)+y]!=-1 &&
               occ_map[((x-1)*num_bins_y)+y-1]!=-1 &&
               occ_map[((x-1)*num_bins_y)+y]!=-1 &&
               occ_map[((x-1)*num_bins_y)+y+1]!=-1 &&
               occ_map[((x+1)*num_bins_y)+y-1]==-1 &&
               occ_map[((x+1)*num_bins_y)+y]==-1 &&
               occ_map[((x+1)*num_bins_y)+y+1]==-1))
        {
          border_map[(x*num_bins_y)+y]=1;
          border_x[*num_bins_border]=x;
          border_y[*num_bins_border]=y;
         // Rprintf("starting at %d %d\n",x,y);
          (*num_bins_border)++;
     //      printf("starting %d at %d %d\n",*num_bins_border,x,y);
          return 1;
        }
        // a border pixels could have 3 non visited pixels above, and none below, or reverse
        if((border_map[((x)*num_bins_y)+y]!=1 &&
           occ_map[((x)*num_bins_y)+y]!=-1 &&
           occ_map[((x-1)*num_bins_y)+y-1]==-1 &&
           occ_map[((x)*num_bins_y)+y-1]==-1 &&
           occ_map[((x+1)*num_bins_y)+y-1]==-1 &&
           occ_map[((x-1)*num_bins_y)+y+1]!=-1 &&
           occ_map[((x)*num_bins_y)+y+1]!=-1 &&
           occ_map[((x+1)*num_bins_y)+y+1]!=-1)
             ||
               (border_map[((x)*num_bins_y)+y]!=1 &&
               occ_map[((x)*num_bins_y)+y]!=-1 &&
               occ_map[((x-1)*num_bins_y)+y-1]!=-1 &&
               occ_map[((x)*num_bins_y)+y-1]!=-1 &&
               occ_map[((x+1)*num_bins_y)+y-1]!=-1 &&
               occ_map[((x-1)*num_bins_y)+y+1]==-1 &&
               occ_map[((x)*num_bins_y)+y+1]==-1 &&
               occ_map[((x+1)*num_bins_y)+y+1]==-1))
        {
          border_map[(x*num_bins_y)+y]=1;
          border_x[*num_bins_border]=x;
          border_y[*num_bins_border]=y;
          (*num_bins_border)++;
        //  printf("starting %d at %d %d\n",*num_bins_border,x,y);
         // Rprintf("starting at %d %d\n",x,y);
          return 1;
        }
        
        // a border pixel could have 3 non visited pixels and be at one of the 4 corners of the rectangle
        // need this otherwise we miss 2 corners in a rectangular map
        if((border_map[((x)*num_bins_y)+y]!=1 && // top left
           occ_map[((x)*num_bins_y)+y]!=-1 &&
           occ_map[((x-1)*num_bins_y)+y]==-1 &&
           occ_map[((x-1)*num_bins_y)+y+1]==-1 &&
           occ_map[((x)*num_bins_y)+y+1]==-1 &&
           occ_map[((x)*num_bins_y)+y-1]!=-1 &&
           occ_map[((x+1)*num_bins_y)+y]!=-1 &&
           occ_map[((x+1)*num_bins_y)+y+1]!=-1)
          ||
            (border_map[((x)*num_bins_y)+y]!=1 && // top right
            occ_map[((x)*num_bins_y)+y]!=-1 &&
            occ_map[((x)*num_bins_y)+y+1]==-1 &&
            occ_map[((x+1)*num_bins_y)+y+1]==-1 &&
            occ_map[((x+1)*num_bins_y)+y]==-1 &&
            occ_map[((x-1)*num_bins_y)+y]!=-1 &&
            occ_map[((x-1)*num_bins_y)+y-1]!=-1 &&
            occ_map[((x)*num_bins_y)+y-1]!=-1)
          ||
            (border_map[((x)*num_bins_y)+y]!=1 && // bottom left
            occ_map[((x)*num_bins_y)+y]!=-1 &&
            occ_map[((x-1)*num_bins_y)+y]==-1 &&
            occ_map[((x-1)*num_bins_y)+y-1]==-1 &&
            occ_map[((x)*num_bins_y)+y-1]==-1 &&
            occ_map[((x)*num_bins_y)+y+1]!=-1 &&
            occ_map[((x+1)*num_bins_y)+y+1]!=-1 &&
            occ_map[((x+1)*num_bins_y)+y]!=-1)
            ||
            (border_map[((x)*num_bins_y)+y]!=1 && // bottom right
            occ_map[((x)*num_bins_y)+y]!=-1 &&
            occ_map[((x)*num_bins_y)+y-1]==-1 &&
            occ_map[((x+1)*num_bins_y)+y-1]==-1 &&
            occ_map[((x+1)*num_bins_y)+y]==-1 &&
            occ_map[((x-1)*num_bins_y)+y]!=-1 &&
            occ_map[((x-1)*num_bins_y)+y+1]!=-1 &&
            occ_map[((x)*num_bins_y)+y+1]!=-1))
        {
          border_map[(x*num_bins_y)+y]=1;
          border_x[*num_bins_border]=x;
          border_y[*num_bins_border]=y;
          (*num_bins_border)++;
          // printf("starting %d at %d %d\n",*num_bins_border,x,y);
         // Rprintf("starting at %d %d\n",x,y);
          return 1;
        }
      }
      else
      {// border could be a visited bin on the edge of the occ map
        if (occ_map[(x*num_bins_y)+y]!=-1&&border_map[((x)*num_bins_y)+y]!=1)
        {
          border_map[(x*num_bins_y)+y]=1;
          border_x[*num_bins_border]=x;
          border_y[*num_bins_border]=y;
          (*num_bins_border)++;
         //  printf("starting %d at %d %d\n",*num_bins_border,x,y);
        //  Rprintf("starting at %d %d\n",x,y);
          return 1;
        }
      }
    }
  }
  return 0;
}

int find_an_adjacent_border_pixel(double* occ_map, int num_bins_x, int num_bins_y,int*border_map,int*border_x,int* border_y,int* num_bins_border)
{
  // look for a pixels around the last added pixel that could be a border, in the 9 pixels around it
  for (int x = border_x[(*num_bins_border)-1]-1;x<=border_x[(*num_bins_border)-1]+1;x++)
  {
    for (int y = border_y[(*num_bins_border)-1]-1;y<=border_y[(*num_bins_border)-1]+1;y++)
    {
      if(
        ((x>0 && // not in x==0, we need to see a -1.0 value on the left side
          x<num_bins_x-1 && // not is x = length(x), we need to see a -1.0 value on the right side
          border_map[((x)*num_bins_y)+y]!=1 &&  // not already considered a border pixel
          occ_map[((x)*num_bins_y)+y]!=-1&&y<num_bins_y) && // bin was visited by animal and is not after y limit
          (occ_map[((x-1)*num_bins_y)+y]==-1 || occ_map[((x+1)*num_bins_y)+y]==-1)) // should have an invalid bin on right or left side
        || 
          ((y>0 && 
          y<num_bins_y-1 && 
          border_map[((x)*num_bins_y)+y]!=1 && 
          occ_map[((x)*num_bins_y)+y]!=-1&&x<num_bins_x) &&
          (occ_map[(x*num_bins_y)+y-1]==-1 || occ_map[(x*num_bins_y)+y+1]==-1)))
      {
        border_map[(x*num_bins_y)+y]=1;
        border_x[*num_bins_border]=x;
        border_y[*num_bins_border]=y;
     //   Rprintf("adding %d %d\n",x,y);
        (*num_bins_border)++;
        find_an_adjacent_border_pixel(occ_map,num_bins_x,num_bins_y,border_map,border_x,border_y,num_bins_border);// recursive search
        return 1;
      }
    }
  }
  return 0;
}
int identify_border_pixels_in_occupancy_map(double* occ_map, int num_bins_x, int num_bins_y,
                                            int* border_map, int* border_x, int* border_y, 
                                            int* num_bins_border)
{
  // set border map to 0
  for(int i = 0; i < num_bins_x*num_bins_y;i++) // set border map to 0
    {border_map[i]=0;}

  *num_bins_border=0;

    while(find_border_starting_point(occ_map, num_bins_x, num_bins_y,border_map,border_x,border_y,num_bins_border))
    {
      //recursive algorhythm! Inspired by conversation with Jozsef and Catherine in Vienna
      find_an_adjacent_border_pixel(occ_map, num_bins_x, num_bins_y,border_map,border_x,border_y,num_bins_border);
    }
  // remove internal bins 
  remove_internal_bins_from_border(num_bins_x, num_bins_y, border_map, border_x, border_y, num_bins_border);
  return 0;
}



int assign_wall_to_border_pixels(int num_bins_x, int num_bins_y, int* border_x, int* border_y, int* num_bins_border,int* wall_id,int* border_map)
{
  /* 
     function assumes that there are 2 vertical and 2 horizontal walls.
     the two vertical or horizontal walls should be in different half of the map
     We simply count the number of border pixels for each row or column of the map
     Walls should results in a high number of pixel for a given row or column
   
     Note that a pixel can only be assigned to one border. It is assigned to the closest wall, if distances are equal the assignation
     is arbitrary set by the order of if statements
  */
  
  double* col_sum = (double*) malloc(num_bins_x*sizeof(double));
  double* row_sum = (double*) malloc(num_bins_y*sizeof(double));
  double sum,max;
  int h1=0,h2=0,v1=0,v2=0; // coordinate of the horizontal and vertical walls
  int dist_h1=0;
  int dist_h2=0;
  int dist_v1=0;
  int dist_v2=0;
  int* x;
  int* y;
  int* id;
  int num_bins_wall;

  for(int i = 0; i < num_bins_x; i++)
  {
    sum=0;
    for(int j = 0; j < *num_bins_border; j++)
    {
      if(border_x[j]==i)
        sum++;
    }
    col_sum[i]=sum;
  }
  
  for(int i = 0; i < num_bins_y; i++)
  {
    sum=0;
    for(int j = 0; j < *num_bins_border; j++)
    {
      if(border_y[j]==i)
        sum++;
    }
    row_sum[i]=sum;
  }
  
  max=0;
  for(int i = 0; i < num_bins_x/2; i++)
  {
    if(col_sum[i]>max)
    {
      max=col_sum[i];
      v1=i;
    }
  }
  max=0;
  for(int i = num_bins_x/2; i < num_bins_x; i++)
  {
    if(col_sum[i]>max)
    {
      max=col_sum[i];
      v2=i;
    }
  }
  max=0;
  for(int i = 0; i < num_bins_y/2; i++)
  {
    if(row_sum[i]>max)
    {
      max=row_sum[i];
      h1=i;
    }
  }
  max=0;
  for(int i = num_bins_y/2; i < num_bins_y; i++)
  {
    if(row_sum[i]>max)
    {
      max=row_sum[i];
      h2=i;
    }
  }
  
  for(int i =0; i < *num_bins_border;i++)
    wall_id[i]=-1;
  
  for(int i =0; i < *num_bins_border;i++)
  {
    dist_v1=sqrt((border_x[i]-v1)*(border_x[i]-v1));
    dist_v2=sqrt((border_x[i]-v2)*(border_x[i]-v2));
    dist_h1=sqrt((border_y[i]-h1)*(border_y[i]-h1));
    dist_h2=sqrt((border_y[i]-h2)*(border_y[i]-h2));
 
 
    // assign to the closest border
    if(dist_v1<2&&dist_v1<=dist_v2&&dist_v1<=dist_h1&&dist_v1<=dist_h2)
       wall_id[i]=0;
    if(dist_v2<2&&dist_v2<=dist_v1&&dist_v2<=dist_h1&&dist_v2<=dist_h2)
      wall_id[i]=1;
    if(dist_h1<2&&dist_h1<=dist_h2&&dist_h1<=dist_v1&&dist_h1<=dist_v2)
      wall_id[i]=2;
    if(dist_h2<2&&dist_h2<=dist_h1&&dist_h2<=dist_v1&&dist_h2<=dist_v2)
      wall_id[i]=3;
  }
  
  // if a pixels is not associated to a wall, remove it from the border_map;
  for(int i =0; i < *num_bins_border;i++)
  {
    if(wall_id[i]==-1)
    {
      border_map[(border_x[i]*num_bins_y)+border_y[i]]=0;
    }
  }
  // remove border pixels that are not close to a wall
  x = (int*)malloc((*num_bins_border)*sizeof(int));
  y = (int*)malloc((*num_bins_border)*sizeof(int));
  id =(int*)malloc((*num_bins_border)*sizeof(int));
  num_bins_wall=0;
  
  for(int i =0; i < *num_bins_border;i++)
  {
    if(wall_id[i]!=-1)
    {
      x[num_bins_wall]=border_x[i];
      y[num_bins_wall]=border_y[i];
      id[num_bins_wall]=wall_id[i];
      num_bins_wall++;
    }
  }
  for(int i = 0; i < num_bins_wall; i++)
  {
    border_x[i]=x[i];
    border_y[i]=y[i];
    wall_id[i]=id[i];
  }
  
  (*num_bins_border)=num_bins_wall;
  
  free(col_sum);
  free(row_sum);
  free(x);
  free(y);
  free(id);
  return 0;
}


void detect_border_pixels_in_occupancy_map(double* occ_map, int* border_map, int num_bins_x, int num_bins_y)
{
    // This function return a border map in which the border pixels have a value of 1 and the rest 0
    
    int num_bins_border=0;
    int total_bins=num_bins_x*num_bins_y;
    int* border_x =  (int*) malloc(total_bins*sizeof(int)); // list of x index for border bins
    int* border_y = (int*) malloc(total_bins*sizeof(int)); // list of y index for border bins
    identify_border_pixels_in_occupancy_map(occ_map, num_bins_x, num_bins_y, border_map, 
                                                border_x, border_y, &num_bins_border);
    
    
    
    free(border_x);
    free(border_y);
}




int find_an_adjacent_field_pixel(double* rate_map, int* field_map, int num_bins_x, int num_bins_y, double threshold, int start_x, int start_y, int* field_pixel_count)
{
    /*
        Recursive function that looks for adjacent pixels that are part of a field.
        
        To be part of a field, the firing rate should be above the threshold.
        
        The function looks at the 9 pixels surrounding rate_map[start_x, start_y], assuming these are valid rates and within the map.
        If a field pixel is found, the function call itself recursively.
        
        field pixels are set to 1 in the field_map
        field pixels are set to -1.0 in the rate_map
        field_pixel_count is increased by 1 each time a pixel is added
    */
  
  for (int x = start_x-1;x<=start_x+1;x++)
  {
    for (int y = start_y-1;y<=start_y+1;y++)
    {
      if( x>=0 && // should be within the map
          x<num_bins_x && // should be within the map
          y>=0 && // should be within the map
          y<num_bins_y && // should be within the map
                
          rate_map[((x)*num_bins_y)+y]!=-1.0 && // not invalid rate
          rate_map[((x)*num_bins_y)+y]> threshold && // above threshold
          field_map[((x)*num_bins_y)+y]== 0) // not already a pixel
      {
        field_map[((x)*num_bins_y)+y]=1; // set the pixels in the field map
        rate_map[((x)*num_bins_y)+y]=-1.0; // set this firing rate to invalid to ensure we won't add it in a different field
        (*field_pixel_count)++;
        find_an_adjacent_field_pixel(rate_map, field_map, num_bins_x, num_bins_y,threshold, x, y,field_pixel_count);// recursive search
      }
    }
  }
  return 0;
}
int detect_one_field(double* rate_map, int* field_map, int num_bins_x, int num_bins_y, double min_peak_rate, double min_peak_rate_proportion)
{
    /*
    Function to detect the field with the largest peak rate
    
    Invalid data in rate_map should be -1.0 and not np.nan
    The function assumes that the field_map is filled with 0.
    
    The pixels that are part of the field will be set to -1.0 in rate_map 
    The pixels that are part of the field will be set to 1 in field_map
    
    The field could be any size, so you might want to filter the results. As long as there is one pixel above min_peak_rate, we will return a field.
    
    
    Arguments
    rate_map: 2D array
    field_map: 2D array
    num_bins_x: size of 2D arrays in x dimension
    num_bins_y: size of 2D arrays in y dimension
    min_peak_rate: minimal peak firing rate to start field detection
    min_peak_rate_proportion: threshold to add pixels to a field, as a proportion of its peak
    
    Returns the number of pixels in the field, and modify rate_map and field_map if there is a field.
    */
    
    double max_rate = 0;
    int field_pixel_count = 0; 
    int peak_x=0;
    int peak_y=0;
    for (int x = 0; x < num_bins_x; x++)
        for (int y = 0; y < num_bins_y; y++)
            if (rate_map[(x*num_bins_y)+y]!=-1.0 && rate_map[(x*num_bins_y)+y] > max_rate){
                max_rate = rate_map[(x*num_bins_y)+y];
                peak_x=x;
                peak_y=y;
            }
    if (max_rate > min_peak_rate){
        field_map[(peak_x*num_bins_y)+peak_y] = 1;
        rate_map[(peak_x*num_bins_y)+peak_y] = -1.0;
        field_pixel_count++;
        double threshold = max_rate * min_peak_rate_proportion;
        
        // this is to prevent the bins surrounding a field with high peak from being detected as a second peak
        if (threshold > min_peak_rate){
            threshold = min_peak_rate;
        }
        
        // loop to add adjacent pixels until all adjacent pixels above the threshold are added.
        // this is a recursive function. It will call itself many times to add adjacent pixels.
        find_an_adjacent_field_pixel(rate_map, field_map, num_bins_x, num_bins_y,threshold, peak_x, peak_y, &field_pixel_count);
    }
    
    return field_pixel_count;
    
}
