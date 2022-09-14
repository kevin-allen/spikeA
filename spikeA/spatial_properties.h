double correlation (double* x, double* y, int size, double invalid);

void map_autocorrelation(double *one_place, // pointer to one place field map
			 double *one_auto, // pointer to one spatial autocorrelation map
			 int x_bins_place_map, // x size of the place field map (num bins)
			 int y_bins_place_map, // y size of the place field map
			 int x_bins_auto_map, // x size of the autocorrelation map
			 int y_bins_auto_map, // y size of the autocorreltion map
			 int min_for_correlation); // minimum of valid values to do the correlation

void remove_internal_bins_from_border(int num_bins_x, int num_bins_y, int* border_map, int* border_x, int* border_y, int* num_bins_border);
int find_border_starting_point(double* occ_map, int num_bins_x, int num_bins_y,int*border_map,int*border_x,int* border_y,int* num_bins_border);
int find_an_adjacent_border_pixel(double* occ_map, int num_bins_x, int num_bins_y,int*border_map,int*border_x,int* border_y,int* num_bins_border);
int identify_border_pixels_in_occupancy_map(double* occ_map, int num_bins_x, int num_bins_y,int* border_map, int* border_x, int* border_y, int* num_bins_border);
int assign_wall_to_border_pixels(int num_bins_x, int num_bins_y, int* border_x, int* border_y, int* num_bins_border,int* wall_id,int* border_map);
void detect_border_pixels_in_occupancy_map(double* occ_map, int* border_map, int num_bins_x, int num_bins_y);
int detect_one_field(double* rate_map, int* field_map, int num_bins_x, int num_bins_y, double min_peak_rate, double min_peak_rate_proportion);
int find_an_adjacent_field_pixel(double* rate_map, int* field_map, int num_bins_x, int num_bins_y, double threshold, int start_x, int start_y,int* field_pixel_count);