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
	hr = HistoryRecorder(max_steps=100) # from POMDPTools
	hist = simulate(hr, tiger, policy, belief_updater, init_dist) # run 100 step simulation
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
	spacecraft_position_y_original::Float64
    spacecraft_position_x::Float64
	spacecraft_position_y::Float64
	spacecraft_radius::Float64
    spacecraft_fuel::Float64
	debris_position_x::Float64
	debris_position_y::Float64
    debris_radius::Float64
end

# ╔═╡ ce359010-f61e-11ea-2f71-a1fc0b6d5300
md"##### Action Space"

# ╔═╡ 5418e8df-383d-4285-b455-f34263f737f3
𝒜 = [-1.,0.,1.] # SpaceInvader moving up, staying, or moving down

# ╔═╡ f2980bc3-e061-476e-b4b1-8dd25d0e8250
md"##### Observation Space"

# ╔═╡ 7926b425-d4d7-4bd4-9d68-ce3c558fdd6b
𝒪 = [-1., 0., 1.] # debris moving left, staying, or moving right

# ╔═╡ 032b2c7b-a8e2-4ca4-b025-ee5a9a888f38
struct DebrisObservationState
    debris_position_x::Float64
	debris_position_y::Float64
    debris_radius::Float64
end

# ╔═╡ 2e6aff30-f61d-11ea-1f71-bb0a7c3aad2e
md"""
##### Initial State
"""

# ╔═╡ 01f1b077-3b8e-4d11-afa8-9ad7a9b26f8c
# define the initial state
s0 = SpaceInvaderState(0.0,5.0,1.0,10.0,10.0,rand((1:10)),1.0)

# ╔═╡ 02937ed0-f4da-11ea-1f82-cb56e99e5e20
initial_state_distr = Deterministic(s0);

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
		
		spacecraft_radius = s.spacecraft_radius
	
		# ------------------------------ Debris State ----------------------------  #
	
		# No change
		debris_position_x = s.debris_position_x - 1
		debris_position_x = clamp(debris_position_x,0.0,10.0) # Boundaries
		
		debris_position_y = s.debris_position_y
		debris_position_y = clamp(debris_position_y,0.0,10.0)
		
		debris_radius = s.debris_radius

		spacecraft_position_y_original = s.spacecraft_position_y_original
	
		# Next state
		sp = SpaceInvaderState(spacecraft_position_y_original, spacecraft_position_x,spacecraft_position_y,spacecraft_radius,spacecraft_fuel,debris_position_x,debris_position_y,debris_radius)
		
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

# ╔═╡ 4b2ef518-2eb2-44c2-90b6-a35652149a22
test_state = SpaceInvaderState(0.0,5.0,1.0,10.0,10.0,4.0,1.0)

# ╔═╡ c2dd6ea7-13d0-41f0-a3d6-bdf4085317e7
# Test if works on example state
s_next = transition_function(test_state,𝒜[1])

# ╔═╡ d00d9b00-f4d7-11ea-3a5c-fdad48fabf71
md"""
### Observation Function
The observation function, or observation model, $O(o \mid s^\prime)$ is given by:

$$P(o_t = o| S_{t'} = s', A_{t} = a)$$

In our case, the observation is already given by the Cunjunction Data Message: **we already have an estimated probability of the debris position in regards with the current spacecraft position: here observations are _normally-distributed noisy_ measurements of the debris position.**
"""

# ╔═╡ f8dcc06d-b217-4166-af1a-bba2c2366947
function observation_function(a, sp::SpaceInvaderState)
	debris_position_x_observed = Normal(sp.debris_position_x,0.1)
	debris_position_y_observed = Normal(sp.debris_position_x,0.05)
   # return Normal(sp.debris_position_x,0.15)
    return Deterministic(sp)
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
	
    # Calculate the distance between the spacecraft and the debris
    distance = sqrt((sp.spacecraft_position_x - sp.debris_position_x)^2 + (sp.spacecraft_position_y - sp.debris_position_y)^2)

	# Rewards computation
    if distance <= s.spacecraft_radius + s.debris_radius
        r += -10.0 # penalize collision
	end
	
	if a == +1 || a == -1
        r += -1.0 # penalize action
    end

	# Reward if the spacecraft stays on the same orbit
	if s.spacecraft_position_y == s.spacecraft_position_y_original
		r+= 5
	end

	# Return final reward
	#println("reward: $r")
	return r
end


# ╔═╡ 480293a6-ba24-40d8-a121-5729ea585d8f
md"""
### Is Terminal
"""

# ╔═╡ 4e92da0e-fc1a-4514-b32e-a485896e66b2
function is_collision(s::SpaceInvaderState)
	
	"""
	"""
    # Calculate the distance between the spacecraft and the debris
    distance = sqrt((s.spacecraft_position_x - s.debris_position_x)^2 + (s.spacecraft_position_y - s.debris_position_y)^2)

	# Rewards computation
    if distance <= s.spacecraft_radius + s.debris_radius
        return true
	else 
		return false
	end
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
	obstype= SpaceInvaderState,
	statetype = SpaceInvaderState,
	
	transition =  transition_function,
	observation = observation_function,
	
	reward = reward_function,
	discount = γ,
	
	initialstate = ImplicitDistribution(rng -> SpaceInvaderState(5.0, 0.0,5.0,1.0,10.0,10.0,rand(1.0:10.0),1.0)),

	isterminal = (s -> s.spacecraft_fuel < 0.5 || is_collision(s))
)

# ╔═╡ 4136d580-53e5-4460-a824-af353be0497a
md"## Solving POMDP"

# ╔═╡ ca05ea9c-bdb9-47a1-9764-6443aecfa877
md"""
### POMCP
"""

# ╔═╡ b51cd4cd-ef29-4475-bd1c-872210b117c7
pomcp_solver = POMCPSolver()

# ╔═╡ 073bd18c-c161-4ef9-9252-ff74cd6b5eb4
spaceinvader_policy = solve(pomcp_solver, spaceinvader_pomdp);

# ╔═╡ b69697d6-7864-4e0f-8449-0227e0b381e1
begin
	using ParticleFilters
	N=10
	
	up = BootstrapFilter(spaceinvader_pomdp, N)
	
	test_policy = FunctionPolicy(b->1)
	b0 = initialstate_distribution(spaceinvader_pomdp)# from POMDPModels
	test_hr = HistoryRecorder(max_steps=30)
	spaceinvader_hist = simulate(test_hr, spaceinvader_pomdp, spaceinvader_policy, up, b0);
end

# ╔═╡ 8c8bf2a5-123c-4e98-98cb-3e2bd42482cf
md"""**Discounted Reward after $(N) steps:**"""

# ╔═╡ 8067ec33-a8b1-4e70-817b-5114f8b88943
println("discounted_reward: $(discounted_reward(spaceinvader_hist))");

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
# TESTS
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
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QMDP = "3aa3ecc9-5a5d-57c8-8188-3e47bd8068d2"
QuickPOMDPs = "8af83fb2-a731-493c-9049-9e19dbce6165"

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
PlutoUI = "~0.7.9"
QMDP = "~0.1.6"
QuickPOMDPs = "~0.2.12"
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

[[CPUTime]]
git-tree-sha1 = "2dcc50ea6a0a1ef6440d6eecd0fe3813e5671f45"
uuid = "a9c8d775-2e2e-55fc-8582-045d282d599e"
version = "1.0.0"

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

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "a243ddf20a9609420716bc1c54443a2678b00c87"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.5.0"

[[HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "2e13c9956c82f5ae8cbdb8335327e63badb8c4ff"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.6.2"

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

[[NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

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

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

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

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

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

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

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

[[UnicodePlots]]
deps = ["Crayons", "Dates", "SparseArrays", "StatsBase"]
git-tree-sha1 = "dc9c7086d41783f14d215ea0ddcca8037a8691e9"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "1.4.0"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
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
# ╠═7926b425-d4d7-4bd4-9d68-ce3c558fdd6b
# ╠═032b2c7b-a8e2-4ca4-b025-ee5a9a888f38
# ╟─2e6aff30-f61d-11ea-1f71-bb0a7c3aad2e
# ╠═01f1b077-3b8e-4d11-afa8-9ad7a9b26f8c
# ╠═02937ed0-f4da-11ea-1f82-cb56e99e5e20
# ╟─eb932850-f4d6-11ea-3102-cbbf0e9d8189
# ╠═4a0f47bb-5b73-4329-9d40-93c9400763a9
# ╠═4b2ef518-2eb2-44c2-90b6-a35652149a22
# ╠═c2dd6ea7-13d0-41f0-a3d6-bdf4085317e7
# ╟─d00d9b00-f4d7-11ea-3a5c-fdad48fabf71
# ╠═f8dcc06d-b217-4166-af1a-bba2c2366947
# ╟─648d16b0-f4d9-11ea-0a53-39c0bfe2b4e1
# ╠═30a63492-a3b9-4516-9883-87f9dc2d5023
# ╟─480293a6-ba24-40d8-a121-5729ea585d8f
# ╠═4e92da0e-fc1a-4514-b32e-a485896e66b2
# ╟─b664c3b0-f52a-11ea-1e44-71034541ace4
# ╠═5cba153b-43e0-43e0-b8e7-14f3f9d337c5
# ╟─b35776ca-6f61-47ee-ab37-48da09bbfb2b
# ╠═d704bda2-98ca-40f0-bb50-9d5b85321376
# ╟─4136d580-53e5-4460-a824-af353be0497a
# ╟─ca05ea9c-bdb9-47a1-9764-6443aecfa877
# ╠═6ccb51c3-3461-4ba1-ac35-19a04dd9e8e6
# ╠═b51cd4cd-ef29-4475-bd1c-872210b117c7
# ╠═073bd18c-c161-4ef9-9252-ff74cd6b5eb4
# ╠═b69697d6-7864-4e0f-8449-0227e0b381e1
# ╟─8c8bf2a5-123c-4e98-98cb-3e2bd42482cf
# ╠═8067ec33-a8b1-4e70-817b-5114f8b88943
# ╟─71406b44-9eed-4e18-b0e8-d1b723d943aa
# ╟─a8b53304-c500-48e8-90ef-40ed362b9a6a
# ╟─b55e63e0-158e-4fa2-bb50-7dad667102ac
# ╟─a35bff87-612c-47a5-b03e-a85f3183cecc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
