-- utility functions for the CNN

-- shuffle the data
function shuffleData(data)
	local newOrder = torch.randperm(data:size(1))
	local shuffle_data = {}
	for i = 1,data:size(1) do
		shuffle_data[i] = data[newOrder[i]]
	end

	return shuffle_data
end

-- select optimization algorithm
function select_optim(optimization)
	local optimState -- the state of the method
	local optimMethod --what method to use
	if optimization == 'lbfgs' then
	  optimState = {
	    learningRate = 1e-1,
	    maxIter = 2,
	    nCorrection = 10
	  }
	  optimMethod = optim.lbfgs
	elseif optimization == 'sgd' then
	  optimState = {
	    learningRate = 1e-2,
	    weightDecay = 0.0005,
	    momentum = 0.9,
	    learningRateDecay = 1e-4
	  }
	  optimMethod = optim.sgd
	elseif optimization == 'adagrad' then
	  optimState = {
	    learningRate = 1e-1,
	  }
	  optimMethod = optim.adagrad
	elseif optimization == 'adadelta' then
	  optimState = {
	  }
	  optimMethod = optim.adadelta
	else
	  error('Unknown optimizer')
	end
	return optimState, optimMethod
end



