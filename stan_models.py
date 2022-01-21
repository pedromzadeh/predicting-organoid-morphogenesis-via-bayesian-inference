import pystan
import pickle

# Build the STAN model
linear_model = """
data {
   int N;
   vector[N] BD;
   vector[N] R;
}
transformed data {
}
parameters {
   real A;
   real B;
   real<lower=0> sigma;
}
model {
   vector[N] bd_exact;

   bd_exact = A * R + B;
   BD ~ normal(bd_exact,sigma);
}
generated quantities {
}
"""

mn_model = """
data {
   int N;                      // number of segments selected for analysis
   int m;                      // number of neighboring segments
   int n_seg;                  // number of total segments
   int n_time;                 // number of frames
   int seg_ids[N];
   matrix[n_time,N] BD;
   matrix[n_time,n_seg] ratio;
}
transformed data {
}
parameters {
   real A;                    // coefficient for this ratio
   real B;                    // offset
   real alpha;                // coefficient for all + ratios
   real beta;                 // coefficient for all - ratios
   real<lower=0> sigma;       // noise in regression
}
model {

   matrix[n_time,N] bd_pred;
   matrix[n_time,2*m+1] all_r;
   vector[n_time] pos_r;
   vector[n_time] neg_r;
   int start_id = 0;
   int end_id = 0;
   int seg_id = -1;

   for (k in 1:N) {
      seg_id = seg_ids[k] + 1;
      start_id = seg_id - m;
      end_id = seg_id + m;
      all_r = ratio[1:n_time,start_id:end_id];

      for (q in 1:n_time) {
         pos_r[q] = 0.0;
         neg_r[q] = 0.0;
      }

      for (q in m+2:2*m+1) {
         pos_r = pos_r + all_r[1:n_time,q];
      }
      for (q in 1:m) {
         neg_r = neg_r + all_r[1:n_time,q];
      }

      bd_pred[1:n_time,k] = B + A * all_r[1:n_time,m+1] + alpha * pos_r + beta * neg_r;
   }

   for (k in 1:N) {
      BD[:,k] ~ normal(bd_pred[:,k],sigma);
   }
   
}
generated quantities {
}
"""

linear_stan_model = pystan.StanModel(model_code = linear_model, model_name = 'linear_model')
mn_stan_model = pystan.StanModel(model_code = mn_model, model_name = 'mn_model')

with open('models.pkl', 'wb') as f:
   pickle.dump({'linear_model' : linear_stan_model, 'mn_model' : mn_stan_model}, f)
