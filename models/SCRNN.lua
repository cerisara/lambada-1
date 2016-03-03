local SCRNN = {}

function SCRNN.scrnn(input_size, rnn_size, context_size, context_scale, hsm)
  
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- 2: prev h
  table.insert(inputs, nn.Identity()()) -- 3: prev c

  -- Normal embedding

  local embedding_layer = nn.LookupTable(input_size, rnn_size)
  local x = embedding_layer(inputs[1])
  local prev_h = inputs[2]
  local prev_c = inputs[3]

  -- Context embedding
  local context_embedding = nn.Sequential()
  context_embedding:add(nn.LookupTable(input_size, context_size))
  context_embedding:add(nn.MulConstant(context_scale))
  local x_c = context_embedding(inputs[1])

  -- Fast projection
  local h2h = nn.LinearNB(rnn_size, rnn_size)(prev_h)

  -- slow projection 
  local c2c = nn.LinearNB(context_size, context_size)(prev_c):annotate{name='c2c'}
  local scaled_c2c = nn.MulConstant(1-context_scale)(c2c)
  local new_c = nn.CAddTable()({x_c, scaled_c2c})

  local c2h = nn.LinearNB(context_size, rnn_size)(new_c)
  local new_h = nn.ReLU()(nn.CAddTable()({x, h2h, c2h}))

  table.insert(outputs, new_h)
  table.insert(outputs, new_c)

  local top_h = nn.JoinTable(2)({new_h, new_c})

  if hsm == false then
    local proj = nn.Linear(rnn_size + context_size, input_size)(top_h)
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)
  else
    -- if HSM is used then softmax will be done later
    table.insert(outputs, top_h)

  end

  return nn.gModule(inputs, outputs)
end

return SCRNN
