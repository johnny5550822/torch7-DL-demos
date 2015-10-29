-- ###############
-- gpu version
-- with optim selection
-- batch
-- This is the main for training a CNN on digit classification

-- ###############

require "torch"
require "nn"
require "math"
require 'xlua' -- provides useful tools, such as progress bars
require 'gnuplot'
require 'cutorch'
require 'paths'
require "optim"
require 'cunn'

-- my own modality
require './model/simple_model.lua' -- one way to incorporate external function

-- parse command line input
local opts = paths.dofile('opts.lua')
params = opts.parse(arg)

-- variables; they must be global variable so that other functions can access it. It's a hack; I don't like this that much,
learningRate = params.lr
maxEpochs = params.me
batch_size = params.bs	
totalImages = 10000 -- we know there are in total 10,000 images; each with size of 1 x 16 x 16
patchSize = 16
maxIterations = totalImages/batch_size -- this is per epochs; I did not calculate the max. total iterations for clarity

-- initiate gpu support
cutorch.setDevice(1) -- gpu device # 1
cutorch.manualSeed(123) -- seed 

-- sequential of ifiles input for different functions
paths.dofile('train.lua')
paths.dofile('test.lua')
paths.dofile('util.lua')

------------------------------------------------------------START
--main 
function main()
	local training_dataset, testing_dataset, classes, classes_names = dofile('data.lua')
	local network = simple_model(#classes)
	s_training_dataset = shuffleData(training_dataset)

    -- determine the optimization method
    local optimState, optimMethod = select_optim(params.optim)
    local parameters, gradParameters = network:getParameters() -- get the parameters of the network

	train_network(network,s_training_dataset, optimMethod, optimState, parameters, gradParameters,testing_dataset, classes, classes_names)
end

--run 
main()


