#### Problem Set 2 - Markov chain process

# Submission by: Guillaume Pousse, Rodrigo Pereira, Morgane Soufflet  
using Distributions, LinearAlgebra


### Part A - Tauchen method ###############################################################
# Input: tauchen function (do not change)
function tauchen(mean, sd, rho, num_states; q=3)

    uncond_sd = sd/sqrt(1-rho^2)
    y = range(-q*uncond_sd, stop = q*uncond_sd, length = num_states)
    d = y[2]-y[1]

    Pi = zeros(num_states,num_states)

    for row = 1:num_states
      # end points
          Pi[row,1] = cdf(Normal(),(y[1] - rho*y[row] + d/2)/sd)
          Pi[row,num_states] = 1 - cdf(Normal(), (y[num_states] - rho*y[row] - d/2)/sd)

      # middle columns
          for col = 2:num_states-1
              Pi[row, col] = (cdf(Normal(),(y[col] - rho*y[row] + d/2) / sd) -
                             cdf(Normal(),(y[col] - rho*y[row] - d/2) / sd))
          end
    end

  yy = y .+ mean # center process around its mean

  Pi = Pi./sum(Pi, dims = 2) # renormalize

  return Pi, yy
end 

# Parameters value
rho = 0.8;
sigma = sqrt(0.1225);
N = 5 ;
N_iter = 2000;
m = 0

## Applying the tauchen method (function below) to get the Markov chain process

Markov_5 = tauchen(m, sigma, rho, N)

# The Markov_5 object is composed of both our probability matrix P and the yy array. 
# We use the collect function to retreive the yy array.

P_1 = Markov_5[1] 
yy = collect(Markov_5[2])

possible_income = transpose(exp.(yy))

## Create a function to get the stable distribution
function get_invdist(guess, tol)
    for iter in 1:N_iter
		new_guess = guess * P_1 # This is our fixed point equation. 
		new_guess = new_guess / sum(new_guess) # We need to normalise our distribution such that it sums to 1
		error = norm(new_guess - guess) # This is the error (or difference) between new_guess and guess. 
        if error < tol
            return new_guess
        elseif iter == N_iter
            error("No solution found after $iter iterations")
        end
        guess = new_guess
    end
end

## Find the stable distribution for the process generated above 
stable_dist_1 = get_invdist(possible_income, 1*10^-10)


## What is the mean income? 

# To find the mean income, we sum every state multiplied by it's probability. 
# By doing so, we build our expected value (or mean value) of income.
mean_income = sum(stable_dist_1[i] * possible_income[i] for i in 1:length(possible_income))


## What is the share of households with income y_5?

highest_income_HH_share = stable_dist_1[end]

# A share of approximately 2.3% of households have the highest income. 





### Part B - Recessions ###############################################################
P_2 = [0.971 0.029 0.000 
    0.145 0.778 0.077
    0.000 0.508 0.492] 


## Assume we are at normal growth, what is the proba of going in recession in t+1? 

# When at normal growth, the probability of going into a mild recession next month is:
P_2[1,2] 

# When at normal growth, the probability of going into a severe recession next month is:
P_2[1,3]

# When at normal growth, the probability of going into a mild or severe recession next month is:
P_2[1,2] + P_2[1,3]

## In t+6? 

# Now, let's compute the probability matrix in 6 months:

P_2_6months = P_2^6

# When at normal growth, the probability of going into a mild recession in 6 months month is:
P_2_6months[1,2]

# When at normal growth, the probability of going into a severe recession in 6 months month is:
P_2_6months[1,3]

# When at normal growth, the probability of going into a mild or severe recession in 6 months is:
P_2_6months[1,2] + P_2_6months[1,3] 


## Compute the stable distribution

# We create a new version of the 'get_invdist' function which uses the P__2 matrix.
function new_get_invdist(guess, tol)
    for iter in 1:N_iter
		new_guess = guess * P_2 # This is our fixed point equation. 
		new_guess = new_guess / sum(new_guess) # We need to normalise our distribution such that it sums to 1. 
		error = norm(new_guess - guess) # This is the error (or difference) between new_guess and guess. 
        if error < tol
            return new_guess
        elseif iter == N_iter
            error("No solution found after $iter iterations")
        end
        guess = new_guess
    end
end

# However, we do not have an initial guess, so we will attribute equal probabilities to all states as a first guess.
equal_guess = [1/3 1/3 1/3] 

# *output = get_invdist(*initial guess*,*something important*)
stable_dist_2 = new_get_invdist(equal_guess, 1*10^-10)

# The probability of being in a severe recession in the long run is:
stable_dist_2[3]