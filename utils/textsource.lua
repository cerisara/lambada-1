--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Sumit Chopra <spchopra@fb.com>
--          Michael Mathieu <myrhev@fb.com>
--          Marc'Aurelio Ranzato <ranzato@fb.com>
--          Tomas Mikolov <tmikolov@fb.com>
--          Armand Joulin <ajoulin@fb.com>


-- This file loads a text dataset.
require 'torch'
require 'paths'
require 'math'
require 'xlua'

local TextSource = torch.class('TextSource')
local preprocessor = require 'utils.preprocessor'

-- config:
-- {
--   threshold : 0
--   shuff : false
--   task : "word"
--   name : "ptb"
--   batch_size : 32
--   nclusters : 0
-- }

function TextSource:__init(config)
    self.batch_size = config.batch_size
    self.root = paths.concat("./data", config.name)
    self.batch_size = config.batch_size
    local clean = config.clean or false

    self.train_txt = paths.concat(self.root, "train.txt")
    self.valid_txt = paths.concat(self.root, "valid.txt")
    self.test_txt = paths.concat(self.root, "test.txt")
    self.vocab_file = paths.concat(self.root, "vocab.txt")
    self.accuracy_valid_txt = paths.concat(self.root, "lambada.txt")
    self.files = {self.train_txt, self.valid_txt, self.test_txt}
    local output_tensors = {}

    if not path.exists(self.vocab_file) then 
        self.vocab_file = nil
    end

    local tensor_file = paths.concat(self.root, "tensors.t7")
    local dict_file = paths.concat(self.root, "dict.t7")
    local lambada_valid_file = paths.concat(self.root, "lambada.t7")

    if ( not path.exists(tensor_file) ) or ( not path.exists(dict_file) ) or clean then
        print("Generating tensor files...")
        self:_build_data(config, dict_file, tensor_file)
    end



    -- get the training, validation and test tensors
    self.sets = {}
    self.dict = torch.load(dict_file)
    local all_data = torch.load(tensor_file)

    -- for i, data in pairs(all_data) do
    self.sets['train'] = all_data[1]
    self.sets['valid'] = all_data[2]
    self.sets['test'] = all_data[3]
    print("Data loaded successfully")
    print("Vocab size: " .. #self.dict.index_to_symbol)

    if path.exists(self.accuracy_valid_txt) then
        if not path.exists(lambada_valid_file) then
            print("Making a list of lambada tensors...")
            local lambada_tensors = preprocessor.text_to_lambada_tensor(self.dict, self.accuracy_valid_txt, config)
            torch.save(lambada_valid_file, lambada_tensors)
        end


        self.lambada_tensors = torch.load(lambada_valid_file)

    end

    collectgarbage()
end

-- return a list of tensor streams
-- a stream is corresponding to one sentence in lambada dataset
function TextSource:get_lambada_streams()

    if self.lambada_tensors == nil then
--        print("lambada tensor is nil")
        return nil
    end
    
    local tensors = self.lambada_tensors

    local reshaped_tensors = {}

    for k, tensor in pairs(tensors) do

        stream_length = tensor:size(1)
        cur_stream = torch.LongTensor(stream_length, 1)
        cur_stream[{{},1}]:copy(tensor[{{1, stream_length}}])

        table.insert(reshaped_tensors, cur_stream)
    end

    return reshaped_tensors
end


function TextSource:create_clusters(config)

    local n_clusters = config.nclusters
    local vocab_size = self:get_vocab_size()

    -- Default: 50 words per clusters (a heuristic though)
    if n_clusters == 0 then
        n_clusters = math.floor(vocab_size / 50)        
    end

    -- Create a Tensor to indicate the clusters with size: vocab_size * 2
    -- Word at index i with have cluster id (clusters[i][1]) and in-cluster id (clusters[i][2])
    self.dict.clusters = torch.LongTensor(vocab_size, 2):zero()
    local n_in_each_cluster = math.ceil(vocab_size / n_clusters) 
    -- Randomly cluster words based on a normal distribution
    local _, idx = torch.sort(torch.randn(vocab_size), 1, true)

    local n_in_cluster = {} --number of tokens in each cluster
    local c = 1
    for i = 1, idx:size(1) do
        local word_idx = idx[i] 
        if n_in_cluster[c] == nil then
            n_in_cluster[c] = 1
        else
            n_in_cluster[c] = n_in_cluster[c] + 1
        end
        self.dict.clusters[word_idx][1] = c
        self.dict.clusters[word_idx][2] = n_in_cluster[c]        
        if n_in_cluster[c] >= n_in_each_cluster then
            c = c + 1
        end
        if c > n_clusters then --take care of some corner cases
            c = n_clusters
        end 
    end
    print(string.format('using hierarchical softmax with %d classes', c))


end

function TextSource:create_frequency_tree(config)

    local bin_size = config.bin_size

    if bin_size <= 0 then
        bin_size = 100
    end

    local sorted_freqs, sorted_indx = torch.sort(self.dict.index_to_freq, true)

    local indices = sorted_indx
    local tree = {}
    local id = indices:size(1)

    function recursive_tree(indices)
      if indices:size(1) < bin_size then
            id = id + 1
            tree[id] = indices
            return
          end
      local parents = {}

      for start = 1, indices:size(1), bin_size do
        local stop = math.min(indices:size(1), start+bin_size-1)
        local bin = indices:narrow(1, start, stop-start+1)
        -- print(bin)
        assert(bin:size(1) <= bin_size)

        id = id + 1
        table.insert(parents, id)

        tree[id] = bin
        -- print(id, bin)
      end
      recursive_tree(indices.new(parents))
    end

    recursive_tree(indices) 

    self.dict.tree = tree
    self.dict.root_id = id

    print('Created a frequency softmaxtree with ' .. self.dict.root_id - indices:size(1) .. ' nodes')

end

function TextSource:get_vocab_size()

    return #self.dict.index_to_symbol
end

function TextSource:_build_data(config, dict_file, tensor_file)

    local output_tensors = {}
    local dict = preprocessor.build_dictionary(config, self.train_txt, self.vocab_file)

    for i, file in pairs(self.files) do
        output_tensors[i] = preprocessor.text_to_tensor(dict, file, config)
    end

    torch.save(dict_file, dict)
    torch.save(tensor_file, output_tensors)
end


-- -- returns the raw data for train|validation|test (given by set_name)
function TextSource:get_set_from_name(set_name)
    local out = self.sets[set_name]
    if out == nil then
        if set_name == 'nil' then
            error('Set name is nil')
        else
            error('Unknown set name: ' .. set_name)
        end
    end
    return out
end


-- This function returns the data corresponding to train|valid|test sets.
-- <sname> is the name of the data type. The data is returned as a 2D tensor
-- of size: (N/batch_size)*batch_size, where N is the number of words.
function TextSource:get_stream(sname)
    local set = self:get_set_from_name(sname)

    local bz = self.batch_size

    -- Only use batch_size of 1 in testing
    if sname == 'test' then
        bz = 1
    end

    -- Cut off the last part of the stream to match with batches
    local stream_length = torch.floor(set:size(1)/bz)
    local cur_stream = torch.LongTensor(stream_length, bz)
    local offset = 1

    -- Create the batches
    for i = 1, bz do
        cur_stream[{{},i}]:copy(set[{{offset,
                                     offset + stream_length - 1}}])
        offset = offset + stream_length
    end
    collectgarbage()
    -- collectgarbage()
    return cur_stream
end
