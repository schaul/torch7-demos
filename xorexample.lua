----------------------------------------------------------------------
-- A tutorial-style script outline for using 
-- torch neural networks on a trivial problem.
--
-- Tom Schaul
----------------------------------------------------------------------


require 'torch'
require 'nn'
require 'nnx'



-- (1) Build a multi-layer neural network

h = 3 -- number of hidden neurons

net = nn.Sequential() -- a sequential processing flow
net:add(nn.Linear(2,h))
net:add(nn.Tanh())
net:add(nn.Linear(h,1))
net:add(nn.Tanh())

-- print the module sequence with corresponding parameters (called "weight" and "bias")
for i,m in ipairs(net.modules) do 
	print(i, m, m:parameters())
end

-- the parameters are initialized randomly, e.g.
ll2 = net.modules[3]
print("weight2", ll2.weight)



-- (2) Create a toy dataset (XOR)
inpt = torch.Tensor(4, 2)
targ = torch.Tensor(4, 1)
for i=1,4 do
	inpt[i][1] = -1+(2*math.floor((i-1)/2))
	inpt[i][2] = -1+2*((i-1)%2)
	targ[i][1] = -inpt[i][1]*inpt[i][2]	
end
print("data", inpt:t()) -- transposed
print("targets", targ:t()) -- transposed



-- (3) Forward/backward mechanism
net:zeroGradParameters()

-- the forward pass takes an input tensor and produces an output (here, a prediction). 
print('predict', net:forward(inpt[1])[1])
-- which is stored in the .output variable
print('again', net.output[1])
-- we can look at internal variables, for example in the second linear layer
print("ll2-out", ll2.output[1])

-- The backward pass takes the same input again, and the output error
-- to produce the error on the input
print("inErr", net:backward(inpt[1], -targ[1]+net.output)[1])

-- Unlike what you might expect, the backward method does not compute
-- the derivatives... for that we call additionally:
print("gBias (still zero)", ll2.gradBias)
net:accGradParameters(inpt[1], -targ[1]+net.output)

-- Now we can look at the corresponding derivatives 
-- stored in the modules, for example:
print("gBias", ll2.gradBias)
print("gWeight", ll2.gradWeight)



-- (4) Define an error function to be minimized (here: mean squared error)
criterion = nn.MSECriterion()

-- the forward of the criterion gives the error value between an output and a target:
e = criterion:forward(net.output, targ[1])
print("Criterion", e, net.output[1], targ[1][1]) 

-- its backward pass tells us what to pass as output error to the network
print("outerr", criterion:backward(net.output, targ[1])) 


function netReset(n) -- helper function that re-randomizes all the net's parameters
	for i, m in pairs(n.modules) do
		if m.reset then m:reset() end
	end
end

function totalMSE() -- sum the criterion-based error for all 4 samples
	local s=0
	for i=1,inpt:size(1) do
		s = s + criterion:forward(net:forward(inpt[i]), targ[i])
	end
	return s
end

-- Inspect the error, for a couple of random parameter initializations
for i=1,5 do
	netReset(net)
	print(totalMSE())
end



-- (5) Training the network
print("init error", totalMSE(), "w", ll2.weight) 

-- by hand for a couple of steps
net:zeroGradParameters() 	
for i=1,4 do
	net:forward(inpt[i])
	-- notice we need to call both "backward" and "accGradParameters".
	net:backward(inpt[i], criterion:backward(net.output, targ[i]))         
	net:accGradParameters(inpt[i], criterion.gradInput)
end
net:updateParameters(0.2) -- this is the learning rate
	
-- the weights and error have changed now
print("1st epoch", totalMSE(), "w",ll2.weight) 


function oneEpoch()
	net:zeroGradParameters() 	
	local s=0
	for i=1,inpt:size(1) do
		s = s + criterion:forward(net:forward(inpt[i]), targ[i])
		net:backward(inpt[i], criterion:backward(net.output, targ[i]))         
		net:accGradParameters(inpt[i], criterion.gradInput)
	end	
	return s
end

-- train more 
for i=1,99 do
	oneEpoch()
	net:updateParameters(0.2)
end
print("100th epoch", totalMSE(),"w", ll2.weight) 



-- (6) Using a trainer and dataset to do the same
netReset(net)
print("new init", totalMSE(), "w",ll2.weight)

-- for this we need to make a dataset object. Simple.
dataset = nn.DataSet() 
for i=1,4 do
	dataset:add{input=inpt[i], output=targ[i]}
end

-- there are many optimizers available, but here is the simplest one
-- (but even SGD has lots of additional parameters that could be set in realistic cases)
optimizer = nn.SGDOptimization{module = net,
                               criterion = criterion,
                               learningRate = 0.2} 

-- if we don't want to guess at the best learning rate we can call this method:
--    optimizer:optimalLearningRate(inpt, targ)
--    print("Optimal learning rate", optimizer.learningRate)
-- It turns out that this does not work in our case (much too big, why?)
                               
-- for some reason, we provide virtually the same information
-- again to construct the trainer
trainer = nn.OnlineTrainer{module = net,
                           criterion = criterion,
                           optimizer = optimizer,
                           maxEpoch = 100,
                           dispProgress=false,
                           }

-- a little hack to silence the trainer's outputs
_print = print; print = function (...) end

-- the actual training happens here.
trainer:train(dataset)
print = _print

-- note that often the weights found in this second run are very different from the ones 
-- in the first run, which is due to the random re-initialization.
print("100th epoch", totalMSE(),"w", ll2.weight) 



-- (7) A third approach is the optimizer framework with "evaluate()" closures
netReset(net)
print("new init", totalMSE(), "w",ll2.weight)

-- For this we need a flattened view on the parameters and derivatives
parameters = nnx.flattenParameters(nnx.getParameters(net))
gradParameters = nnx.flattenParameters(nnx.getGradParameters(net))

-- For example, using the LBFGS algorithm
require 'liblbfgs'

-- this needs to be a closure that does a forward-backward 
lbfgs.evaluate = oneEpoch

-- init LBFGS state
maxIterations = 100
maxLineSearch = 10
maxEvaluation = 100
linesearch = 0 
sparsity = 0
verbose = 0
lbfgs.init(parameters, gradParameters,
           maxEvaluation, maxIterations, maxLineSearch,
           sparsity, linesearch, verbose)

-- here's where the optimization takes place
output = lbfgs.run()

print("100th epoch", totalMSE(),"w", ll2.weight) 






