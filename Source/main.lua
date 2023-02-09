import "CoreLibs/graphics"
import "CoreLibs/sprites"

-- import "util"
-- import "engine"

local gfx <const> = playdate.graphics

-- UI/driver state
local turn = 1
local cursorLoc = {x=1, y=1}
local gameRunning = true

local board = board:new()

local function drawStone(x, y, color)
  local radius <const> = 8
  if color==1 then -- white=0, black=1
    gfx.fillCircleAtPoint(x, y, radius)
  else
    -- need to clear the grid underneath
    gfx.setColor(gfx.kColorWhite)
    gfx.fillCircleAtPoint(x, y, radius)
    gfx.setColor(gfx.kColorBlack)
    gfx.drawCircleAtPoint(x, y, radius)
  end
end

local function drawGrid()
  -- display size is 400x240
  -- grid is 8x8

  -- the grid needs to be evenly spaced in x and y direction
  -- so max size it can be is 240px
  -- but that'd be kinda bulky, so let's do 75% = 180px
  --
  -- so we're doing 180x180

  local margin_x <const>  = (400-180)/2
  local margin_y <const>  = (240-180)/2

  local cell_width <const> = 180/8 -- fractional, but that's okay

  -- do vertical lines
  local line_x           = margin_x
  local line_y <const>   = margin_y
  local line_len <const> = 240-margin_y*2

  for i=1, 9 do
    gfx.drawLine(line_x, line_y, line_x, line_y+line_len)
    line_x += cell_width
  end

  -- do horizontal lines
  local line_x <const> = margin_x
  local line_y         = margin_y
  local line_len       = 400-margin_x*2

  for i=1, 9 do
    gfx.drawLine(line_x, line_y, line_x+line_len, line_y)
    line_y += cell_width
  end

  function drawStoneGridCoord(x,y,color)
    local center_x <const> = margin_x + x*cell_width - cell_width/2
    local center_y <const> = margin_y + y*cell_width - cell_width/2
    drawStone(center_x, center_y, color)
  end

  for x=1, 8 do
    for y=1, 8 do
      local cell = board:get_cell(x,y)
      if cell ~= nil then
        drawStoneGridCoord(x, y, cell)
      end
    end
  end

  -- FIXME draw possible moves?

  function drawCursor(x, y) -- FIXME this isn't quite centered
    local line_x <const> = margin_x + x*cell_width + 2
    local line_y <const> = margin_y + y*cell_width + 2
    gfx.drawLine(line_x, line_y, line_x+18, line_y+18)

    local line_x <const> = margin_x + x*cell_width + 2
    local line_y <const> = margin_y + y*cell_width + cell_width - 2
    gfx.drawLine(line_x, line_y, line_x+18, line_y-18)
  end

  drawCursor(cursorLoc.x-1, cursorLoc.y-1)

  gfx.drawText("NEXT", 5, 5)
  drawStone(50, 13, turn)
end

local function drawGameOver()
  gfx.clear()

  local cntWhite = 0
  local cntBlack = 0

  for x=1, 8 do
    for y=1, 8 do
      local cell = board:get_cell(x,y)
      if cell ~= nil then
        if cell == 0 then
          cntWhite = cntWhite + 1
        else
          cntBlack = cntBlack + 1
        end
      end
    end
  end

  if cntWhite > cntBlack then -- is tie possible?
    gfx.drawText("WHITE wins", 5, 5)
  else
    gfx.drawText("BLACK wins", 5, 5)
  end

  playdate.display.flush()
end

local function redraw()
  gfx.clear()
  drawGrid()
  playdate.display.flush()
end

-- all drawing will be event driven
playdate.stop()
redraw()

local goInputHandlers = {
  upButtonUp = function()
    if gameRunning then
      cursorLoc.y = cursorLoc.y - 1
      redraw()
    end
  end,

  downButtonUp = function()
    if gameRunning then
      cursorLoc.y = cursorLoc.y + 1
      redraw()
    end
  end,

  leftButtonUp = function()
    if gameRunning then
      cursorLoc.x = cursorLoc.x - 1
      redraw()
    end
  end,

  rightButtonUp = function()
    if gameRunning then
      cursorLoc.x = cursorLoc.x + 1
      redraw()
    end
  end,

  AButtonUp = function()
    if not gameRunning then
      board = board:new()
      gameRunning = true
      turn = 1

      redraw()
      return
    end

    if board:make_move(cursorLoc.x, cursorLoc.y, turn) ~= nil then
      local movesForWhite <const> = board:can_move(0) ~= nil
      local movesForBlack <const> = board:can_move(1) ~= nil

      if turn==0 then -- white
        if board:can_move(1) == 1 then
          turn = 1
        else
          turn = 0
        end
      else -- turn=1
        if board:can_move(0) == 1 then
          turn = 0
        else
          turn = 1
        end
      end

      if board:can_move(turn) ~= 1 then
        gameRunning = false
        drawGameOver()
      else
        redraw()
      end
    end
  end,
}

playdate.inputHandlers.push(goInputHandlers)
