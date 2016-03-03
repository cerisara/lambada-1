local HLogSoftMax, parent = torch.class('nn.HLogSoftMax', 'nn.Criterion')
local HSMClass = require('models.HSMClass')


function HLogSoftMax:__init(mapping, input_size, sizeAverage)
    -- different implementation of the fbnn.HSM module
    -- variable names are mostly the same as in fbnn.HSM
    -- only supports batch inputs

    self.sizeAverage = sizeAverage or true
    parent.__init(self)
    if type(mapping) == 'table' then
         self.mapping = torch.LongTensor(mapping)
    else
         self.mapping = mapping
    end
    self.input_size = input_size
    self.n_classes = self.mapping:size(1)
    self.n_clusters = self.mapping[{{},1}]:max()
    self.n_class_in_cluster = torch.LongTensor(self.n_clusters):zero()
    self.n_max_class_in_cluster = self.mapping[{{},2}]:max()


    self.reverse_mapping = torch.LongTensor(self.n_clusters, self.n_max_class_in_cluster):zero()

    for i = 1, self.mapping:size(1) do
        local c = self.mapping[i][1]
        self.n_class_in_cluster[c] = self.n_class_in_cluster[c] + 1

        local cidx = self.mapping[i][2]

        self.reverse_mapping[c][cidx] = i
    end
    
    
    --cluster softmax/loss
    self.cluster_model = nn.Sequential()
    self.cluster_model:add(nn.Linear(input_size, self.n_clusters))
    self.cluster_model:add(nn.LogSoftMax())
    self.logLossCluster = nn.ClassNLLCriterion(nil, self.sizeAverage)
    
    --class softmax/loss
    self.class_model = HSMClass.hsm(self.input_size, self.n_clusters, self.n_max_class_in_cluster)
    local get_layer = function (layer)
		          if layer.name ~= nil then
			      if layer.name == 'class_bias' then
			          self.class_bias = layer
			      elseif layer.name == 'class_weight' then
                                  self.class_weight = layer
                              end
		          end    
		      end
    self.class_model:apply(get_layer)
    self.logLossClass = nn.ClassNLLCriterion(nil, self.sizeAverage)
    -- self.logLossClass.sizeAverage = self.sizeAverage

    self:change_bias()
    self.gradInput = torch.Tensor(input_size)

    self.full_probs = torch.FloatTensor(self.n_classes):zero()

end

function HLogSoftMax:clone(...)
    return nn.Module.clone(self, ...)
end

function HLogSoftMax:parameters()
    return {self.cluster_model.modules[1].weight,
            self.cluster_model.modules[1].bias,
            self.class_bias.weight,
            self.class_weight.weight} ,
           {self.cluster_model.modules[1].gradWeight,
            self.cluster_model.modules[1].gradBias,
            self.class_bias.gradWeight,
            self.class_weight.gradWeight}
end

function HLogSoftMax:getParameters()
    return nn.Module.getParameters(self)
end


-- A function that aims to compute the full distribution
-- Only supports batch size 1
-- This function should be used in CPU mode though (very slow for GPU)
function HLogSoftMax:generateDistribution(input, target)
    assert(input:size(1) == 1, 'full distribution only supports batch 1')

    -- A 2D tensor storing probs for all nodes in the HSM
    -- local lossTable = torch.FloatTensor(self.n_clusters, self.n_max_class_in_cluster)
    -- lossTable:zero()
    -- allocate memory 
    self.full_probs:zero()

    -- Fast indexing 
    local map_target = self.mapping[target[1]]
    local c_target = map_target[1]
    local cidx_target = map_target[2]
    -- Distribution at cluster levels
    local cluster_dist = self.cluster_model:forward(input)

    local label = torch.LongTensor(1):zero()
    local word
    local loss = 0

    -- Compute the distribution at class levels
    for i = 1, self.n_clusters do
        
        label[1] = i
        local dist = self.class_model:forward{input, label}  


        for j = 1, self.n_max_class_in_cluster do

            word = self.reverse_mapping[i][j]
            
            -- negative log likelihood from cluster and class
            -- for batch 1 it is much faster than calling the ClassNLLCriterion
            if word > 0 then -- some values may do not have any reverse mapping

                self.full_probs[word] = - cluster_dist[1][i] - dist[1][j]

                if c_target == i and cidx_target == j then
                    loss = self.full_probs[word]
                end

            end
        end
    end

    -- FOR DEBUGGING: double check the log prob computed by 2 functions

    -- target = target:long()
    -- local loss2 = self:updateOutput(input, target)
    -- print(loss, loss2)

    -- For some precision problem, two losses can be different (maybe due to the underlying C implementation)
    -- But in testing the difference is negligable
    -- -- assert(loss == loss2)

    return self.full_probs, loss
end


function HLogSoftMax:updateOutput(input, target)
    local batch_size = input:size(1)
    if torch.type(target) ~= 'torch.CudaTensor' then
        target = target:long()
    end
    local new_target = self.mapping:index(1, target)

    local cluster_target = new_target:select(2, 1)
    local cluster_loss = self.logLossCluster:forward(
                   self.cluster_model:forward(input),
                   cluster_target)

    local class_target = new_target:select(2, 2)
    local class_loss = self.logLossClass:forward(
                        self.class_model:forward({input, cluster_target}),
                        class_target)
    self.output = cluster_loss + class_loss
    
    local n_valid = 1

    if self.sizeAverage == false then
        n_valid = batch_size
    end

    return self.output, batch_size                   
end

function HLogSoftMax:updateGradInput(input, target)
    self.gradInput:resizeAs(input)

    -- Avoid CPU error
    if torch.type(target) ~= 'torch.CudaTensor' then
        target = target:long()
    end

    local new_target = self.mapping:index(1, target)
    -- backprop clusters
    self.logLossCluster:updateGradInput(self.cluster_model.output,
                                        new_target:select(2,1))    
    self.gradInput:copy(self.cluster_model:backward(input,
                        self.logLossCluster.gradInput))
    -- backprop classes
    self.logLossClass:updateGradInput(self.class_model.output,
                                      new_target:select(2,2))
    self.gradInput:add(self.class_model:backward(input,
                       self.logLossClass.gradInput)[1])
    return self.gradInput
end


function HLogSoftMax:backward(input, target, scale)
    self:updateGradInput(input, target)
    return self.gradInput
end

function HLogSoftMax:change_bias()
    -- hacky way to deal with variable cluster sizes
    for i = 1, self.n_clusters do
        local c = self.n_class_in_cluster[i]
        for j = c+1, self.n_max_class_in_cluster do
            self.class_bias.weight[i][j] = math.log(0)
        end        
    end
end

