
__kernel void training( const int Sdim, const int Ndim, const int Mdim, const float beta, 
		__global float* IS, __global float* W, 
		__global float* eucl_dist,__local float* ISwrk)
{

	int k,j;
	int i = get_global_id(0);
	
	int iloc = get_local_id(0);
	int nloc = get_local_size(0); 
    	int nRd = Mdim/nloc;           
       
	float tmp;
	if( (i < Ndim) ){

		for(j=0; j < Sdim; j++){           // jth column of IS is braught to local memory as Bwrk.
		
			for(k = iloc * nRd; k < (iloc + 1) *nRd ;k++){        
				ISwrk[k] = IS[ j * Mdim + k];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		
			tmp = 0.0;                             
		
			for( k = 0; k < Mdim ; k++){
				tmp += (ISwrk[k] - W[i*Mdim +k]) * (ISwrk[k] - W[i*Mdim +k]);   // distance of weight vector with ip vector
			}
		
			eucl_dist[i] = tmp;                 //global array of size N contains dist of each neuron
			
			barrier(CLK_GLOBAL_MEM_FENCE);

			int winner_idx = 0;
			
			int min = 64;
			
			for (k = 0; k < Ndim;k++)
			{
				if (min > eucl_dist[k])
				{
					min = eucl_dist[k];
					winner_idx = k;
				}
			}

			barrier(CLK_GLOBAL_MEM_FENCE);

			float t_dist = abs(abs(i - winner_idx) - Ndim/2);    
			float neighValue = beta/pow(2,t_dist);

			for (k = 0; k< Mdim ; k++)                                   // Update the weight vector
			{
				W[i*Mdim + k] = W[i*Mdim + k] - neighValue * (W[i*Mdim + k] - ISwrk[k]) ;
			}
			
			barrier(CLK_GLOBAL_MEM_FENCE); //write-after-read for Bwrk
		}
	}		
}
