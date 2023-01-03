### A Pluto.jl notebook ###
# v0.19.17

using Markdown
using InteractiveUtils

# ╔═╡ 746961b0-f4b6-11ea-3289-03b36dffbea7
begin
	using PlutoUI

	md"""
	# Spacecraft Collision Avoidance
	##### A Partially Observable Markov Decision Process

	A model of Spacecraft Collision Avoidance using POMDPs.jl
	"""
end

# ╔═╡ d0a58780-f4d2-11ea-155d-f55c848f91a8
using POMDPs, QuickPOMDPs, POMDPModelTools, BeliefUpdaters,Distributions, Parameters, POMDPModels, POMDPSimulators,QMDP, POMDPPolicies

# ╔═╡ 6ccb51c3-3461-4ba1-ac35-19a04dd9e8e6
using BasicPOMCP

# ╔═╡ cbed0721-ec2d-400c-b115-202c733cfc68
using Plots, Reel

# ╔═╡ a88c0bf0-f4c0-11ea-0e61-853ac9a0c0cb
md"## Partially Observable MDP (POMDP)"

# ╔═╡ 32c56c10-f4d2-11ea-3c79-3dc8b852c182
md"""
A partially observable Markov decision process (POMDP) is a 7-tuple consisting of:

$\langle \mathcal{S}, \mathcal{A}, {\color{blue}\mathcal{O}}, T, R, {\color{blue}O}, \gamma \rangle$

Variable           | Description          | `POMDPs` Interface
:----------------- | :------------------- | :-------------------:
$\mathcal{S}$      | State space          | `POMDPs.states`
$\mathcal{A}$      | Action space         | `POMDPs.actions`
$\mathcal{O}$      | Observation space    | `POMDPs.observations`
$T$                | Transition function  | `POMDPs.transision`
$R$                | Reward function      | `POMDPs.reward`
$O$                | Observation function | `POMDPs.observation`
$\gamma \in [0,1]$ | Discount factor      | `POMDPs.discount`

Notice the addition of the observation space $\mathcal{O}$ and observation function $O$, which differ from the MDP 5-tuple.

Indeed, the agent receives _observations_ of the current state rather than the true state—and using past observations, it builds a _belief_ of the underlying state (this can be represented as a probability distribution over true states).
"""

# ╔═╡ a13e36e0-f4d2-11ea-28cf-d18a43e34c3e
md"### Set of State variables $\mathcal{S}$
The *state* set in our problem is composed of both $satellite$ and $debris$ respective state *variables*.

**Orbit location $satellite$**

-  $a_{sat}$: size of semi-major axis of $orbit_{sat}$
-  $e_{sat}$: eccentricity of $orbit_{sat}$
-  $\omega_{sat}:$ argument of $orbit_{sat}$ (angle from the ascending node radius to the periapsis radius in a counter clockwise direction)
-  $\nu_{sat}:$ position of $orbit_{sat}$ (angle from the periapsis radius to the object)

⇒ gives the **position of the satellite** on a **2D orbit plane**

**Orbit location $debris$**

-  $a_{deb}$: size of semi-major axis of $orbit_{deb}$
-  $e_{deb}$: eccentricity of $orbit_{deb}$
-  $\omega_{deb}:$ argument of $orbit_{deb}$ (angle from the ascending node radius to the periapsis radius in a counter clockwise direction)
-  $\nu_{deb}:$ position of $orbit_{deb}$ (angle from the periapsis radius to the object

⇒ gives the **position of the debris** on a **2D orbit plane**

**Speed**

-  $V_{sat}$
-  $V_{deb}$

**Advisory**

-  $s_{adv}$: current advisory

**Other parameters**

-  $s_{fuel}$: fuel level

$$\begin{align}
\mathcal{S_{sat}} &= \{\ a_{sat},e_{sat},\omega_{sat},\nu_{sat},V_{sat},s_{adv},s_{fuel}\}\\
\mathcal{S_{deb}} &= \{\ a_{deb},e_{deb},\omega_{deb},\nu_{deb},V_{deb}\}
\end{align}$$
"

# ╔═╡ dbc77e50-f529-11ea-0d79-71196165ac17


# ╔═╡ 222c7568-d9b3-4148-9800-c42372969c14
md"""### Set of Actions $\mathcal{A}$
The *actions* set in our problem is composed of clear $advisories$ to be given as **instructions** to the satellite.

$$\begin{align}
\mathcal{A} = \{\rm &Clear\ of\ Conflict,\\
&Monitor\ Speed\ V\,\\
&Accelerate,\\
&Decelerate\}\\

\end{align}$$
"""

# ╔═╡ d6416ab2-f080-487e-8e36-6c46afddaba2
md"""
## Data available: Cunjunction Data Messages
"""

# ╔═╡ 84d8bf8a-db6f-4eb7-aaff-4b1559066cc7
md"""
### Context
Today, active collision avoidance among orbiting satellites has become a routine task in space operations, relying on validated, accurate and timely space surveillance data.  

For a typical satellite in Low Earth Orbit, hundreds of alerts are issued every week corresponding to possible close encounters between a satellite and another space object (in the form of conjunction data messages CDMs).  

After automatic processing and filtering, there remain about 2 actionable alerts per spacecraft and week, requiring detailed follow-up by an analyst. On average, at the European Space Agency, more than one collision avoidance manoeuvre is performed per satellite and year.
"""

# ╔═╡ 0785e176-7228-4fea-a111-418b9d43f5ab
md"""
### What is a Cunjunction Data Messages ?

As of estimations done in January 2019, more than 34,000 objects with a size larger than 10cm are orbiting our planet. Of these, 22,300 are tracked by the Space Surveillance Network and their position released in the form of a globally shared catalogue.

ESA's Space Debris Office supports collision avoidance activities covering the ESA missions Aeolus, Cryosat-2 and the constellation of Swarm-A/B/C in low-Earth orbit and Cluster-II in highly eccentric orbit approaching the Geostationary (GEO) region. On top of these, more than a dozen spacecraft of partner agencies and commercial operators are supported.

![alt text](https://kelvins.esa.int/media/public/ckeditor_uploads/2021/08/05/new_swarm.png)

In the context of this activity, the orbits of these satellites are propagated and when a close approach with any object in the catalogue is detected a Conjunction Data Message (CDM) is assembled and released. Each CDM contains multiple attributes about the approach, such as the identity of the satellite in question, the object type of the potential collider, the time of closest approach (TCA), the uncertainty (i.e. covariances), etc. It also contains a self-reported risk, which is computed using some of the attributes from the CDM. In the days following the first CDM, as the uncertainties of the objects positions become smaller, other CDMs are released refining the knowledge acquired on the close encounter.

Typically, a time series of CDMs covering one week is released for each unique close approach, with about 3 CDMs becoming available per day. For a given close approach the last obtained CDM, including the computed risk, can be assumed to be the best knowledge we have about the potential collision and the state of the two objects in question. In most cases, the Space Debris Office will alarm control teams and start thinking about a potential avoidance manoeuvre 2 days prior to the close approach in order to avoid the risk of collision, to then make a final decision 1 day prior. In this challenge, we ask to build a model that makes use of the CDMs recorded up to 2 days prior to the closest approach to predict the final risk (i.e. the risk predicted in the last available CDM prior to close approach).

More about the dataset used in this competition and the attributes contained in the various CDMs can be found in the data section. You can also learn some more about the current way ESA's Space Debris office deals with collision avoidance manoeuvres reading this paper.

We thank the US Space Surveillance Network for the provision of surveillance data supporting safe operations of ESA’s spacecraft. Specifically, we are grateful to the agreement which allows to publicly release the dataset for the purpose of this competition.
"""

# ╔═╡ 04cd2d6c-a457-46a4-9bdc-ecc793030989
md"""
### At what frequency is a CDM sent ?

Typically, a time series of CDMs covering one week is released for each unique close approach, with about **3 CDMs becoming available per day**. For a given close approach the **last obtained CDM**, including the **computed risk**, can be assumed to be the **best knowledge** we have **about the potential collision** and the **state of the two objects in question**. 

### What time window to maneuver ?

In most cases, the Space Debris Office will alarm control teams and start thinking about a **potential avoidance manoeuvre _2 days_ prior** to the close approach in order to avoid the risk of collision, to then make a **final decision _1 day_ prior**.
"""

# ╔═╡ 33b27b26-2b32-4620-9212-262fb30fcbbd
md"## Julia Model"

# ╔═╡ 56de9cff-f43b-48d4-ae2d-5e2a4a34cb7c
md"""
### Test Example: TigerPOMDP
"""

# ╔═╡ 5b29d570-cc22-4075-918d-f4fba6969b48
begin
	# initialize problem and solver
	tiger = TigerPOMDP() # from POMDPModels
	solver = QMDPSolver() # from QMDP
	
	# compute a policy
	policy = solve(solver, tiger)
	
	#evaluate the policy
	belief_updater = updater(policy) # the default QMDP belief updater (discrete Bayesian filter)
	init_dist = initialstate_distribution(tiger) # from POMDPModels
	tiger_hr = HistoryRecorder(max_steps=100) # from POMDPTools
	hist = simulate(tiger_hr, tiger, policy, belief_updater, init_dist) # run 100 step simulation
	println("reward: $(discounted_reward(hist))")
end

# ╔═╡ be1258b0-f4db-11ea-390e-2bcc849111d0
md"""
### State, Action, and Observation Spaces
"""

# ╔═╡ 9df137d0-f61c-11ea-0dd6-67535f3b0d52
md"We define our state, action, and observation *spaces*."

# ╔═╡ c720f8a0-f61e-11ea-155d-c13361437a85
md"##### State Space"

# ╔═╡ 115b9d60-1cae-498d-a447-adeaa2269523
# Define state space
struct SpaceInvaderState
    spacecraft_position_x::Float64
	spacecraft_position_y::Float64
	spacecraft_position_y_original::Float64
	spacecraft_radius::Float64
    spacecraft_fuel::Float64
	debris_position_x::Float64
	debris_position_y::Float64
    debris_radius::Float64
end

# ╔═╡ ce359010-f61e-11ea-2f71-a1fc0b6d5300
md"##### Action Space"

# ╔═╡ 5418e8df-383d-4285-b455-f34263f737f3
𝒜 = [-1.,0.,1.] # SpaceInvader moving down, staying, or moving up

# ╔═╡ f2980bc3-e061-476e-b4b1-8dd25d0e8250
md"##### Observation Space"

# ╔═╡ 032b2c7b-a8e2-4ca4-b025-ee5a9a888f38
struct DebrisObservationState
    debris_position_x::Float64
	debris_position_y::Float64
    debris_radius::Float64
end

# ╔═╡ eb932850-f4d6-11ea-3102-cbbf0e9d8189
md"""
### 2.3 Transition Function
"""

# ╔═╡ 4a0f47bb-5b73-4329-9d40-93c9400763a9
function transition_function(s::SpaceInvaderState, a)
	ImplicitDistribution() do rng
		# ---------------------- Satellite State ---------------------------  #
		
		# Update the position_y based on the action and add noise
		spacecraft_position_y = s.spacecraft_position_y + a + randn(rng) * 0.1
		spacecraft_position_y = clamp(spacecraft_position_y,0.0,10.0) #Boundaries
		
		# Update the fuel level
		if a == +1 || a == -1
			spacecraft_fuel = s.spacecraft_fuel - 1
		else
			spacecraft_fuel = s.spacecraft_fuel
		end
		
		# No change
		spacecraft_position_x = s.spacecraft_position_x
		spacecraft_position_x = clamp(spacecraft_position_x,0.0,10.0) #Boundaries

		spacecraft_position_y_original = s.spacecraft_position_y_original
		
		spacecraft_radius = s.spacecraft_radius
	
		# ------------------------------ Debris State ----------------------------  #
	
		# No change
		debris_position_x = s.debris_position_x - 1
		debris_position_x = clamp(debris_position_x,0.0,10.0) # Boundaries
		
		debris_position_y = s.debris_position_y
		debris_position_y = clamp(debris_position_y,0.0,10.0)
		
		debris_radius = s.debris_radius
	
		# Next state
		sp = SpaceInvaderState(spacecraft_position_x,spacecraft_position_y,spacecraft_position_y_original,spacecraft_radius,spacecraft_fuel,debris_position_x,debris_position_y,debris_radius)
		
		# Return the transition probability
		#println("action $a")
		#println("-> spacecraft_position_x $spacecraft_position_x")
		#println("-> spacecraft_position_y $spacecraft_position_y")
		#println("-> spacecraft_radius $spacecraft_radius")
		#println("-> spacecraft_fuel $spacecraft_fuel")
		#println("-> debris_position_x $debris_position_x")
		#println("-> debris_position_y $debris_position_y")
		#println("-> debris_radius $debris_radius")

		return sp
	end
end

# ╔═╡ d00d9b00-f4d7-11ea-3a5c-fdad48fabf71
md"""
### Observation Function
The observation function, or observation model, $O(o \mid s^\prime)$ is given by:

$$P(o_t = o| S_{t'} = s', A_{t} = a)$$

In our case, the observation is already given by the Cunjunction Data Message: **we already have an estimated probability of the debris position in regards with the current spacecraft position: here observations are _normally-distributed noisy_ measurements of the debris position.**
"""

# ╔═╡ f8dcc06d-b217-4166-af1a-bba2c2366947
function observation_function(a, sp::SpaceInvaderState)
	ImplicitDistribution() do rng
		debris_position_x_observed = sp.debris_position_x + randn(rng)
		debris_position_y_observed = sp.debris_position_y + randn(rng)
		debris_radius_observed = sp.debris_radius + randn(rng)
		
		obs = SpaceInvaderState(sp.spacecraft_position_x,sp.spacecraft_position_y,sp.spacecraft_position_y_original,sp.spacecraft_radius,sp.spacecraft_fuel,debris_position_x_observed,debris_position_y_observed,debris_radius_observed)

		return obs
	end
	# return Normal(sp.debris_position_y,0.15)
	# return Deterministic(sp)
end

# ╔═╡ 9301940c-2e8a-47d8-b921-05555d661a8d
function space_invader_observation_function(a, s::SpaceInvaderState)
    # Calculate observation mean
    mean = [s.debris_position_x, s.debris_position_y, s.debris_radius]
    
    # Calculate observation covariance
    covariance = [
        0.1 0.0 0.0
        0.0 0.1 0.0
        0.0 0.0 0.1
    ]
    
    return MvNormal(mean, covariance)
end


# ╔═╡ 777f7f8a-29d6-431d-94f0-6df954d6c747
md"
### Terminal state
"

# ╔═╡ dbf6353a-b39a-44cd-ab88-76e468f48651
function iscollision(s::SpaceInvaderState)
	"""
	Calculates the distance between the spacecraft and the debris using the x and y coordinates of the spacecraft and the debris, and the radii of the spacecraft and the debris. If the distance is smaller than the sum of the radii of the spacecraft and the debris, the function returns true, indicating that there is a collision. Otherwise, it returns false, indicating that there is no collision.
	"""
	# Instanciate r
	r = 0.0
	
    # Calculate the distance between the spacecraft and the debris
    distance = sqrt((s.spacecraft_position_x - s.debris_position_x)^2 + (s.spacecraft_position_y - s.debris_position_y)^2)

	# Rewards computation
    if distance <= s.spacecraft_radius + s.debris_radius
        return true
	else
		return false
	end
end

# ╔═╡ 648d16b0-f4d9-11ea-0a53-39c0bfe2b4e1
md"""
### Reward Function
The reward function is addative, meaning we get a reward of $r_\text{danger}$ whenever the spacecraft is in danger *plus* $r_\text{accelerate}$ whenever we accelerate the spacecraft or *plus* $r_\text{decelerate}$ whenever we decelerate the spacecraft.
"""

# ╔═╡ 30a63492-a3b9-4516-9883-87f9dc2d5023
function reward_function(s::SpaceInvaderState, a, sp::SpaceInvaderState)
	"""
	Calculates the distance between the spacecraft and the debris using the x and y coordinates of the spacecraft and the debris, and the radii of the spacecraft and the debris. If the distance is smaller than the sum of the radii of the spacecraft and the debris, the function returns true, indicating that there is a collision. Otherwise, it returns false, indicating that there is no collision.
	"""
	# Instanciate r
	r = 0.0

	# Rewards computation
    if iscollision(s)
        r += -10.0 # penalize collision
	end
	
	if a == +1 || a == -1
        r += -1.0 # penalize action
    end

	if sp.spacecraft_position_y == s.spacecraft_position_y_original
		r += 2 # reward spacecraft when returning to original orbit
	end

	# Return final reward
	# println("reward: $r")
	return r
end


# ╔═╡ b664c3b0-f52a-11ea-1e44-71034541ace4
md"
### Discount Factor
For an infinite horizon problem, we set the discount factor $\gamma \in [0,1]$ to a value where $\gamma < 1$ to discount future rewards.
"

# ╔═╡ 5cba153b-43e0-43e0-b8e7-14f3f9d337c5
γ = 0.9

# ╔═╡ b35776ca-6f61-47ee-ab37-48da09bbfb2b
md"""
### POMDP Structure using `QuickPOMDPs`
We again using `QuickPOMDPs.jl` to succinctly instantiate the Spacecraft Collision Avoidance POMDP.
"""

# ╔═╡ d704bda2-98ca-40f0-bb50-9d5b85321376
spaceinvader_pomdp = QuickPOMDP(
	actions  = 𝒜,
	obstype= Vector{Float64},
	statetype = SpaceInvaderState,
	
	transition =  transition_function,
	observation = space_invader_observation_function,
	
	reward = reward_function,
	discount = γ,
	
	initialstate = ImplicitDistribution(rng -> SpaceInvaderState(0.0,5.0,5.0,1.0,10.0,10.0,rand(1.:10.),1.0)),

	isterminal = (s -> s.spacecraft_fuel < 0.5 || iscollision(s))
)

# ╔═╡ 4136d580-53e5-4460-a824-af353be0497a
md"## Solving POMDP"

# ╔═╡ ca05ea9c-bdb9-47a1-9764-6443aecfa877
md"""
### POMCP
"""

# ╔═╡ bf2c8ca6-4172-4f48-a6d6-55602e947edb
md"""##### Planner Instanciation """

# ╔═╡ b51cd4cd-ef29-4475-bd1c-872210b117c7
pomcp_solver = POMCPSolver()

# ╔═╡ 073bd18c-c161-4ef9-9252-ff74cd6b5eb4
spaceinvader_policy = solve(pomcp_solver, spaceinvader_pomdp);

# ╔═╡ b69697d6-7864-4e0f-8449-0227e0b381e1
begin

	# Particle Filter: The BootstrapFilter function handles the process of updating the belief state at each time step based on the observations and actions => no need to explicitly specify a belief updater function, as this is handled by the filter.
	using ParticleFilters
	N=10
	filter = BootstrapFilter(spaceinvader_pomdp, N) 
	
	b0 = initialstate_distribution(spaceinvader_pomdp) # from POMDPModels
	
	hr = HistoryRecorder(max_steps=20)
	
	spaceinvader_history = simulate(hr, spaceinvader_pomdp, spaceinvader_policy, filter, b0);
end

# ╔═╡ cefe3ea1-28db-4bb5-aec5-0e471f6b9359
md"""##### Simulation"""

# ╔═╡ 8c8bf2a5-123c-4e98-98cb-3e2bd42482cf
md""" ##### Discounted Reward after $(N) steps"""

# ╔═╡ 8067ec33-a8b1-4e70-817b-5114f8b88943
println("discounted_reward: $(discounted_reward(spaceinvader_history))");

# ╔═╡ bb03e4da-4330-4e38-abf3-63e83ec48fdc
begin
	test_belief = Deterministic(SpaceInvaderState(0.0,5.0,5.0,1.0,10.0,4.0,5.0,1.0))
	a,info = action_info(spaceinvader_policy, test_belief)
end	

# ╔═╡ 4855a10a-6a9b-42e3-9283-accd4d1996d5
md"""##### Vizualisation"""

# ╔═╡ f2f14339-e2b4-4b01-a3cf-d24d4389967e
begin
	frames = Frames(MIME("image/png"), fps=4)
	for b in belief_hist(spaceinvader_history)
	    local ys = [s.debris_position_y for s in particles(b)]
	    local nbins = max(1, round(Int, (maximum(ys)-minimum(ys))*2))
	    push!(frames, histogram(ys,
	                            xlim=(0,10),
	                            ylim=(0,10),
	                            nbins=nbins,
	                            label="",
	                            title="Particle Histogram")
	                            
	    )
	end
	write("hist.gif", frames)
end

# ╔═╡ 71406b44-9eed-4e18-b0e8-d1b723d943aa
md"""
## Concise POMDP definition

```julia
using POMDPs, POMDPModelTools, QuickPOMDPs

@enum State danger safe
@enum Action clearofconflict accelerate decelerate
@enum Observation CDM NoCDM

pomdp = QuickPOMDP(
    states       = [danger, safe],  # 𝒮
    actions      = [clearofconflict, accelerate, decelerate],  # 𝒜
    observations = [CDM, NoCDM], # 𝒪
    initialstate = [safe],          # Deterministic initial state
    discount     = 0.9,             # γ

    transition = function T(s, a)
        if a == accelerate
            return SparseCat([danger, safe], [0, 1])

		elseif a == decelerate
            return SparseCat([danger, safe], [0, 1])

		elseif s == danger && a == clearofconflict
            return SparseCat([danger, safe], [1, 0])

		elseif s == safe && a == clearofconflict
            return SparseCat([danger, safe], [0.1, 0.9])
        end
    end,

    observation = function O(s, a, s′)
        if s′ == danger
            return SparseCat([CDM, NoCDM], [0.9, 0.1])
		elseif s′ == safe
            return SparseCat([CDM, NoCDM], [0.1, 0.9])
        end
    end,

    reward = (s,a)->(s == danger ? -10 : 0) + (a == accelerate ? -5 : 0) + (a == 		decelerate ? -5 : 0)
)

# Solve POMDP
using QMDP
solver = QMDPSolver()
policy = solve(solver, pomdp)

# Query policy for an action, given a belief vector
𝐛 = [0.2, 0.8]
a = action(policy, 𝐛)
```
"""

# ╔═╡ a8b53304-c500-48e8-90ef-40ed362b9a6a
md"""
---
"""

# ╔═╡ b55e63e0-158e-4fa2-bb50-7dad667102ac
md"""
# ToDo
"""

# ╔═╡ 274c41a1-63d2-4f9f-9b91-06123d3d7787
# ------- Done ------- #
# Survivre jusqu'a la fin de la mission du satellite 
# Faire reward tant qu'on reste sur le bon orbit

# ------- En cours ------- #
# Plus travailler sur l'observation et décriptage du CDM belief

# ------- Not Started ------- #
# Cas ou plusieurs débris et choix d"aller vers celui le plus ou moins sur ? 
# Ajouter éviter puis revenir à la position initiale en fonction du belief

# ╔═╡ 5e0d37c5-89d4-427c-9edd-112151193c44
md"""
# Q&A
"""

# ╔═╡ e63db197-83cf-4831-a0e1-6748c7baf033
md"""
- **How are the weights updated for each particle in the particle collection ?**
In particle filtering, the weights of each particle are updated based on the current observation. Specifically, each particle is associated with a weight that reflects its relevance to the current observation. To update the weights, the probability of the current observation based on the state of each particle is calculated, and this probability is used as a weighting factor for the particle's weight.

Here, the following function updates the weights in a particle filtering algorithm using the **probability density function of the observation (p_obs)** and the **current weights (w)** of the particles:

```julia
num_particles = length(w)
for i in 1:num_particles
  w[i] = w[i] * p_obs(y, x[i])
end
```

Here, `y` is the current observation, `x[i]` is the state of particle `i`, and `w[i]` is its weight. The function p_obs calculates the probability of the observation y based on the state `x[i]`. The weight of each particle is updated by multiplying its current weight by the probability of the current observation based on the state of the particle.
"""

# ╔═╡ 49bd4f30-c8a1-42c2-954d-2ac3ee9c6213
md"""
- **What is the difference between the number of tree queries and the number of particles ?**

In POMCP (Partially Observable Monte Carlo Planning), the number of tree_queries refers to the number of queries that the algorithm makes to the tree during a single iteration of the planning process, while the number of particles refers to the number of simulated histories or "particles" that the algorithm maintains to represent the distribution over states in the partially observable environment.

Here is a brief summary of the difference between these two concepts:

- **Number of tree queries**: This determines the number of queries that the algorithm makes to the tree at each iteration of the planning process. The number of tree queries can affect the speed and accuracy of the planning process.

- **Number of particles**: This determines the number of simulated histories or "particles" that the algorithm maintains to represent the distribution over states in the partially observable environment. The number of particles can affect the accuracy of the planning process, as a larger number of particles can better capture the uncertainty in the environment.

In summary, the number of tree queries determines how much information the algorithm gathers from the tree at each iteration, while the number of particles determines how well the algorithm can represent the uncertainty in the environment. Both of these factors can influence the accuracy and efficiency of the planning process.
"""

# ╔═╡ 24af0536-ed89-40cb-889f-4a43e685998a
md"""
- **Covariance matrix**

A 3-dimensional covariance matrix is defined as a 3x3 matrix that contains the covariances between each pair of dimensions of the observation. It is usually denoted by Sigma (Σ) and can be written as follows:

$$\Sigma =
\begin{bmatrix}
\sigma_{11} & \sigma_{12} & \sigma_{13} \\
\sigma_{21} & \sigma_{22} & \sigma_{23} \\
\sigma_{31} & \sigma_{32} & \sigma_{33}
\end{bmatrix}$$

In this formula, Sigma(ij) is the covariance between the i-th and j-th dimensions of the observation. For example, Sigma(12) is the covariance between the 1st and 2nd dimensions of the observation.

It is important to note that the diagonal of the covariance matrix (i.e. the elements Sigma(11), Sigma(22), and Sigma(33)) contain the variances of each dimension of the observation, while the elements off the diagonal (i.e. the elements Sigma(12), Sigma(13), Sigma(21), Sigma(23), and Sigma(31)) contain the covariances between the dimensions.
"""

# ╔═╡ a35bff87-612c-47a5-b03e-a85f3183cecc
html"""
<script>
var section = 0;
var subsection = 0;
var headers = document.querySelectorAll('h2, h3');
for (var i=0; i < headers.length; i++) {
    var header = headers[i];
    var text = header.innerText;
    var original = header.getAttribute("text-original");
    if (original === null) {
        // Save original header text
        header.setAttribute("text-original", text);
    } else {
        // Replace with original text before adding section number
        text = header.getAttribute("text-original");
    }
    var numbering = "";
    switch (header.tagName) {
        case 'H2':
            section += 1;
            numbering = section + ".";
            subsection = 0;
            break;
        case 'H3':
            subsection += 1;
            numbering = section + "." + subsection;
            break;
    }
    header.innerText = numbering + " " + text;
};
</script>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BasicPOMCP = "d721219e-3fc6-5570-a8ef-e5402f47c49e"
BeliefUpdaters = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
POMDPModelTools = "08074719-1b2a-587c-a292-00f91cc44415"
POMDPModels = "355abbd5-f08e-5560-ac9e-8b5f2592a0ca"
POMDPPolicies = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
POMDPSimulators = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
ParticleFilters = "c8b314e2-9260-5cf8-ae76-3be7461ca6d0"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QMDP = "3aa3ecc9-5a5d-57c8-8188-3e47bd8068d2"
QuickPOMDPs = "8af83fb2-a731-493c-9049-9e19dbce6165"
Reel = "71555da5-176e-5e73-a222-aebc6c6e4f2f"

[compat]
BasicPOMCP = "~0.3.8"
BeliefUpdaters = "~0.2.3"
Distributions = "~0.25.45"
POMDPModelTools = "~0.3.13"
POMDPModels = "~0.4.16"
POMDPPolicies = "~0.4.3"
POMDPSimulators = "~0.3.14"
POMDPs = "~0.9.5"
Parameters = "~0.12.3"
ParticleFilters = "~0.5.3"
Plots = "~1.38.0"
PlutoUI = "~0.7.9"
QMDP = "~0.1.6"
QuickPOMDPs = "~0.2.12"
Reel = "~1.3.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractTrees]]
git-tree-sha1 = "52b3b436f8f73133d7bc3a6c71ee7ed6ab2ab754"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.3"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BasicPOMCP]]
deps = ["CPUTime", "Colors", "D3Trees", "MCTS", "POMDPLinter", "POMDPTools", "POMDPs", "Parameters", "ParticleFilters", "Printf", "Random"]
git-tree-sha1 = "14af46b5e2ef3030443ff17bb70c41996850b8c1"
uuid = "d721219e-3fc6-5570-a8ef-e5402f47c49e"
version = "0.3.8"

[[BeliefUpdaters]]
deps = ["POMDPTools", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "8819a9a0e9e9002125ae55626e10f0c210959c30"
uuid = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
version = "0.2.3"

[[BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CPUTime]]
git-tree-sha1 = "2dcc50ea6a0a1ef6440d6eecd0fe3813e5671f45"
uuid = "a9c8d775-2e2e-55fc-8582-045d282d599e"
version = "1.0.0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4ce9393e871aca86cc457d9f66976c3da6902ea7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.4.0"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "d853e57661ba3a57abcdaa201f4c9917a93487a2"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.4"

[[Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[D3Trees]]
deps = ["AbstractTrees", "HTTP", "JSON", "Random", "Sockets"]
git-tree-sha1 = "cace6d05f71aeefe7ffd6f955a0725271f2b6cd5"
uuid = "e3df1716-f71e-5df9-9e2d-98e193103c45"
version = "0.3.3"

[[DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[DiscreteValueIteration]]
deps = ["POMDPLinter", "POMDPTools", "POMDPs", "Printf", "SparseArrays"]
git-tree-sha1 = "62d78a713948c4a95df289ca0eb8639697e1d2eb"
uuid = "4b033969-44f6-5439-a48b-c11fa3648068"
version = "0.4.6"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "24d26ca2197c158304ab2329af074fbe14c988e4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.45"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "bcc737c4c3afc86f3bbc55eb1b9fabcee4ff2d81"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.2"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "64ef06fa8f814ff0d09ac31454f784c488e22b29"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.2+0"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "a243ddf20a9609420716bc1c54443a2678b00c87"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.5.0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "2e13c9956c82f5ae8cbdb8335327e63badb8c4ff"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.6.2"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[MCTS]]
deps = ["Colors", "D3Trees", "POMDPLinter", "POMDPTools", "POMDPs", "Printf", "ProgressMeter", "Random"]
git-tree-sha1 = "48f7a1f54843f18a98b6dc6cd2edba9db70bdcb8"
uuid = "e12ccd36-dcad-5f33-8774-9175229e7b33"
version = "0.5.1"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "df6830e37943c7aaa10023471ca47fb3065cc3c4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.2"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6e9dba33f9f2c44e08a020b0caf6903be540004"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.19+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[POMDPModelTools]]
deps = ["CommonRLInterface", "Distributions", "LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs", "Random", "Reexport", "SparseArrays", "Statistics", "Tricks", "UnicodePlots"]
git-tree-sha1 = "36d32d62e036ae8ebb9b8efe9e8658f902815700"
uuid = "08074719-1b2a-587c-a292-00f91cc44415"
version = "0.3.13"

[[POMDPModels]]
deps = ["ColorSchemes", "Compose", "Distributions", "POMDPTools", "POMDPs", "Parameters", "Printf", "Random", "StaticArrays", "StatsBase"]
git-tree-sha1 = "289a869e7a4816fc353e8a292328056a497e4efd"
uuid = "355abbd5-f08e-5560-ac9e-8b5f2592a0ca"
version = "0.4.16"

[[POMDPPolicies]]
deps = ["Distributions", "LinearAlgebra", "POMDPTools", "POMDPs", "Parameters", "Random", "Reexport", "SparseArrays", "StatsBase"]
git-tree-sha1 = "bd72fbfea89a64946963518aa53097e2a1233c59"
uuid = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
version = "0.4.3"

[[POMDPSimulators]]
deps = ["DataFrames", "Distributed", "NamedTupleTools", "POMDPLinter", "POMDPTools", "POMDPs", "ProgressMeter", "Random", "Reexport"]
git-tree-sha1 = "3735b7a48bd892f153ab7327cb71e447e8f18e14"
uuid = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
version = "0.3.14"

[[POMDPTesting]]
deps = ["POMDPTools", "POMDPs", "Random", "Reexport"]
git-tree-sha1 = "e5acfa4a9c84252491860939a2af7f4ffe501057"
uuid = "92e6a534-49c2-5324-9027-86e3c861ab81"
version = "0.2.6"

[[POMDPTools]]
deps = ["CommonRLInterface", "DataFrames", "Distributed", "Distributions", "LinearAlgebra", "NamedTupleTools", "POMDPLinter", "POMDPs", "Parameters", "ProgressMeter", "Random", "Reexport", "SparseArrays", "Statistics", "StatsBase", "Tricks", "UnicodePlots"]
git-tree-sha1 = "ef73c26402974cd51b9bd395ba5d95a4e34c1b37"
uuid = "7588e00f-9cae-40de-98dc-e0c70c48cdd7"
version = "0.1.2"

[[POMDPs]]
deps = ["Distributions", "Graphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "9ab2df9294d0b23def1e5274a0ebf691adc8f782"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.5"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[ParticleFilters]]
deps = ["POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "9cdc1db2a4992d1ba19bf896372b4eaaac78fa98"
uuid = "c8b314e2-9260-5cf8-ae76-3be7461ca6d0"
version = "0.5.3"

[[Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "5b7690dd212e026bbab1860016a6601cb077ab66"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.2"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "513084afca53c9af3491c94224997768b9af37e8"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.0"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[QMDP]]
deps = ["DiscreteValueIteration", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "Random"]
git-tree-sha1 = "4f5b20454c103900dbd6aa74184c16d311a5063c"
uuid = "3aa3ecc9-5a5d-57c8-8188-3e47bd8068d2"
version = "0.1.6"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[QuickPOMDPs]]
deps = ["BeliefUpdaters", "NamedTupleTools", "POMDPModelTools", "POMDPTesting", "POMDPs", "Random", "Tricks", "UUIDs"]
git-tree-sha1 = "53b35c8174e56a24d350c66e10ec3ce141530e0c"
uuid = "8af83fb2-a731-493c-9049-9e19dbce6165"
version = "0.2.12"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "18c35ed630d7229c5584b945641a73ca83fb5213"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[Reel]]
deps = ["FFMPEG"]
git-tree-sha1 = "0f600c38899603d9667111176eb6b5b33c80781e"
uuid = "71555da5-176e-5e73-a222-aebc6c6e4f2f"
version = "1.3.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "e4bdc63f5c6d62e80eb1c0043fcc0360d5950ff7"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.10"

[[Tricks]]
git-tree-sha1 = "ae44af2ce751434f5fa52e23f46533b45f0cfd81"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.5"

[[URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[UnicodePlots]]
deps = ["Crayons", "Dates", "SparseArrays", "StatsBase"]
git-tree-sha1 = "dc9c7086d41783f14d215ea0ddcca8037a8691e9"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "1.4.0"

[[Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─746961b0-f4b6-11ea-3289-03b36dffbea7
# ╟─a88c0bf0-f4c0-11ea-0e61-853ac9a0c0cb
# ╟─32c56c10-f4d2-11ea-3c79-3dc8b852c182
# ╟─a13e36e0-f4d2-11ea-28cf-d18a43e34c3e
# ╟─dbc77e50-f529-11ea-0d79-71196165ac17
# ╟─222c7568-d9b3-4148-9800-c42372969c14
# ╟─d6416ab2-f080-487e-8e36-6c46afddaba2
# ╟─84d8bf8a-db6f-4eb7-aaff-4b1559066cc7
# ╟─0785e176-7228-4fea-a111-418b9d43f5ab
# ╟─04cd2d6c-a457-46a4-9bdc-ecc793030989
# ╟─33b27b26-2b32-4620-9212-262fb30fcbbd
# ╠═d0a58780-f4d2-11ea-155d-f55c848f91a8
# ╟─56de9cff-f43b-48d4-ae2d-5e2a4a34cb7c
# ╠═5b29d570-cc22-4075-918d-f4fba6969b48
# ╟─be1258b0-f4db-11ea-390e-2bcc849111d0
# ╟─9df137d0-f61c-11ea-0dd6-67535f3b0d52
# ╟─c720f8a0-f61e-11ea-155d-c13361437a85
# ╠═115b9d60-1cae-498d-a447-adeaa2269523
# ╟─ce359010-f61e-11ea-2f71-a1fc0b6d5300
# ╠═5418e8df-383d-4285-b455-f34263f737f3
# ╟─f2980bc3-e061-476e-b4b1-8dd25d0e8250
# ╠═032b2c7b-a8e2-4ca4-b025-ee5a9a888f38
# ╟─eb932850-f4d6-11ea-3102-cbbf0e9d8189
# ╠═4a0f47bb-5b73-4329-9d40-93c9400763a9
# ╟─d00d9b00-f4d7-11ea-3a5c-fdad48fabf71
# ╠═f8dcc06d-b217-4166-af1a-bba2c2366947
# ╠═9301940c-2e8a-47d8-b921-05555d661a8d
# ╟─777f7f8a-29d6-431d-94f0-6df954d6c747
# ╠═dbf6353a-b39a-44cd-ab88-76e468f48651
# ╟─648d16b0-f4d9-11ea-0a53-39c0bfe2b4e1
# ╠═30a63492-a3b9-4516-9883-87f9dc2d5023
# ╟─b664c3b0-f52a-11ea-1e44-71034541ace4
# ╠═5cba153b-43e0-43e0-b8e7-14f3f9d337c5
# ╟─b35776ca-6f61-47ee-ab37-48da09bbfb2b
# ╠═d704bda2-98ca-40f0-bb50-9d5b85321376
# ╟─4136d580-53e5-4460-a824-af353be0497a
# ╟─ca05ea9c-bdb9-47a1-9764-6443aecfa877
# ╠═6ccb51c3-3461-4ba1-ac35-19a04dd9e8e6
# ╟─bf2c8ca6-4172-4f48-a6d6-55602e947edb
# ╠═b51cd4cd-ef29-4475-bd1c-872210b117c7
# ╠═073bd18c-c161-4ef9-9252-ff74cd6b5eb4
# ╟─cefe3ea1-28db-4bb5-aec5-0e471f6b9359
# ╠═b69697d6-7864-4e0f-8449-0227e0b381e1
# ╟─8c8bf2a5-123c-4e98-98cb-3e2bd42482cf
# ╠═8067ec33-a8b1-4e70-817b-5114f8b88943
# ╠═bb03e4da-4330-4e38-abf3-63e83ec48fdc
# ╟─4855a10a-6a9b-42e3-9283-accd4d1996d5
# ╠═cbed0721-ec2d-400c-b115-202c733cfc68
# ╠═f2f14339-e2b4-4b01-a3cf-d24d4389967e
# ╟─71406b44-9eed-4e18-b0e8-d1b723d943aa
# ╟─a8b53304-c500-48e8-90ef-40ed362b9a6a
# ╟─b55e63e0-158e-4fa2-bb50-7dad667102ac
# ╠═274c41a1-63d2-4f9f-9b91-06123d3d7787
# ╟─5e0d37c5-89d4-427c-9edd-112151193c44
# ╟─e63db197-83cf-4831-a0e1-6748c7baf033
# ╟─49bd4f30-c8a1-42c2-954d-2ac3ee9c6213
# ╟─24af0536-ed89-40cb-889f-4a43e685998a
# ╟─a35bff87-612c-47a5-b03e-a85f3183cecc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
