-- for testing the network
local mistakes = 0
local tested_samples = 0
local test_loss = 0 -- calculate the loss for testing example

function test_predictor(predictor, criterion, test_dataset, classes, classes_names)
        predictor:evaluate()
        
        -- loop
        for i=1,test_dataset:size() do

               local input  = test_dataset[i][1]
               local class_id = test_dataset[i][2]

               --cuda
               input = input:float():cuda()
               local label = torch.Tensor(1):fill(class_id):float():cuda()
        		
        	   -- calculate the probability	
               local responses_per_class  =  predictor:forward(input) 
               local probabilites_per_class = torch.exp(responses_per_class)
               local probability, prediction = torch.max(probabilites_per_class, 1) 

               -- update loss
               test_loss = test_loss + criterion:forward(predictor:forward(input),label)

                      
               -- finding mismatch
               if prediction[1] ~= class_id then
                      mistakes = mistakes + 1
                      -- local label = classes_names[ classes[class_id] ]
                      -- local predicted_label = classes_names[ classes[prediction[1] ] ]
                      -- print(i , label , predicted_label )
               end

               tested_samples = tested_samples + 1
        end

        -- update
        local test_err =  mistakes/tested_samples
        test_loss = test_loss / tested_samples -- get the average loss
        return test_err,test_loss
end
