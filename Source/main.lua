import "CoreLibs/graphics"
import "CoreLibs/sprites"

-- import "util"
-- import "engine"

local gfx <const> = playdate.graphics

-- UI/driver state
local N = 8
local turn = 0
local cursorLoc = {x=1, y=1}

local board = board:new()

local function drawStone(x, y, color)
  local radius <const> = 5
  if color==0 then -- white=0, black=1
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

  local sep_x <const>  = 400/(N+N/2)
  local sep_y <const>  = (240-40)/N

  -- do vertical lines
  local line_x         = sep_x*(N/4)
  local line_y <const> = 20
  local line_len <const> = (N-1)*sep_y

  for i=1, N do
    gfx.drawLine(line_x, line_y, line_x, line_y+line_len)
    line_x += sep_x
  end

  function drawStoneOnGrid(x,y,color)
    x = sep_x*(x-1) + sep_x*(N/4)
    y = sep_y*(y-1) + 20
    drawStone(x,y,color)
  end

  -- do horizontal lines
  local line_x <const> = sep_x*(N/4)
  local line_len       = (N-1)*sep_x
  local line_y         = 20

  for i=1, N do
    gfx.drawLine(line_x, line_y, line_x+line_len, line_y)
    line_y += sep_y
  end

  for x=1, N do
    for y=1, N do
      local cell = board:get_cell(x,y)
      if cell ~= nil then
        drawStoneOnGrid(x, y, cell)
      end
    end
  end

  -- draw the cursor
  gfx.drawRect((cursorLoc.x-1) * sep_x + sep_x*(N/4) - 6,
               (cursorLoc.y-1) * sep_y + 20 - 6,
               12, 12)

  gfx.drawText("NEXT", 5, 5)
  drawStone(50, 13, turn)
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
    cursorLoc.y = cursorLoc.y - 1
    redraw()
  end,

  downButtonUp = function()
    cursorLoc.y = cursorLoc.y + 1
    redraw()
  end,

  leftButtonUp = function()
    cursorLoc.x = cursorLoc.x - 1
    redraw()
  end,

  rightButtonUp = function()
    cursorLoc.x = cursorLoc.x + 1
    redraw()
  end,

  AButtonUp = function()
    if board:make_move(cursorLoc.x, cursorLoc.y, turn) ~= nil then
      if turn==0 then
        turn = 1
      else
        turn = 0
      end

      redraw()
    end
  end,
}

playdate.inputHandlers.push(goInputHandlers)
