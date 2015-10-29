-- train; heavily rely on the main variable declaration

-- initiate the confusion matrix to store tp, tn, etc
local train_confusion = optim.ConfusionMatrix(10) 

function train_network(network,dataset, optimMethod, optimState, parameters, gradParameters, testing_dataset, classes, classes_names)
	print('Training the network......')
	local criterion = nn.ClassNLLCriterion():cuda()

	----------------------------- this is the evaluation function for the optimization
	function feval(x)
		-- check if x is updated to date		
		if x ~= parameters then	
			parameters:copy(x)	-- parameters are first defined in the main()
		end

		-- prepare for the batch
		startPos = batch_size * (batch_counter-1) + 1
		endPos = batch_size * batch_counter
		input = torch.Tensor(batch_size,1,patchSize,patchSize):zero() -- this iteration batch
		target = torch.Tensor(batch_size):zero()
		k = 1 -- internal counter for the batch
		for i=startPos,endPos do
			input[{{k},1,{},{}}]:copy(dataset[i][1])
			target[k] = dataset[i][2]
			k = k + 1
		end
		batch_counter = batch_counter + 1 -- update batch_counter

		-- cuda
		input = input:float():cuda()
		target = target:float():cuda()

		-- set gradient parameters to zero
		gradParameters:zero()

		------------------- compute loss and gradient
		-- forward propagation
		local batch_outputs = network:forward(input)
		local batch_loss = criterion:forward(batch_outputs,target)

		-- backward propagation
		local dloss_doutput = criterion:backward(batch_outputs,target)
		network:backward(input, dloss_doutput)

		-- update confusion matrix
		train_confusion:batchAdd(batch_outputs,target)

		return batch_loss, gradParameters
	end

	-----------------------training 
	local train_losses = {} -- training losses for each epoch
	local test_losses = {} -- testing losses
	local test_errs = {} -- testing error
	for epoch = 1, maxEpochs do
		-- batch processing parameters
    	batch_counter = 1
    	print(string.format('Epoch No:%d. (Max=%d)',epoch,maxEpochs))

    	-- maxIterations = 10
		for iteration= 1,maxIterations do
			xlua.progress(iteration,maxIterations) -- progress bar

			local _,minibatch_loss = optimMethod(feval,parameters, optimState)

			-- print loss at certain time; don't have to print all the time
			if iteration%10 ==0 then
				print('Training loss:'..minibatch_loss[1])
			end

			-- update losses for each epoch
			if iteration == maxIterations then
				train_losses[#train_losses + 1] = minibatch_loss[1]

				-- print
				train_confusion:updateValids()
				print('Training confusion matrix')
				print(train_confusion)
				print('Training accuracy:' .. train_confusion.totalValid * 100)
			end
		end
		-- reset confusion matrix
		train_confusion:zero()

		-- testing
		test_err, test_loss = test_predictor(network,criterion, testing_dataset, classes, classes_names)
		test_errs[#test_errs +1] = test_err
		test_losses[#test_losses + 1] = test_loss

		-- print		        
        print( "----------------------" )
        print( "Index Label Prediction" )
        print("Test loss:" .. test_loss)
       	print("Test error:" .. test_err)
       	print("Test accuracy:" .. (1-test_err)*100)
        print( "----------------------" )

		-- collect the garbage in case; update on 10/28/15 no need to do collect garbage anymore according to the official site
		-- collectgarbage()

		-- for each epoch, store the models
		if (epoch%5 ==0 or epoch == maxEpochs) then
			torch.save(string.format('trained_model/%s-%d-epochs.model',model_prefix,epoch),network)
		end

	end

	---------------------------------- once this training is done, plot 
	-- the losses
	gnuplot.plot({
	  'train-loss',
	  torch.range(1, #train_losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
	  torch.Tensor(train_losses),           -- y-coordinates (the training losses)
	  '-'},
	  {
	  'test-loss',
	  torch.range(1, #test_losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
	  torch.Tensor(test_losses),           -- y-coordinates (the training losses)
	  '-'}
	 )
  	gnuplot.title('Training error')
	gnuplot.xlabel('Number of epochs')
	gnuplot.ylabel('Loss')

	-- the error
	gnuplot.figure()
	gnuplot.plot({
	  'test-error',
	  torch.range(1, #test_errs),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
	  torch.Tensor(test_errs),           -- y-coordinates (the training losses)
	  '-'}
	 )
  	gnuplot.title('Classification Error')
	gnuplot.xlabel('Number of epochs')
	gnuplot.ylabel('Error')
end