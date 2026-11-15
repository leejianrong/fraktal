class_name Drone
extends Node3D
## A drone (FRK-93): flies at any height, in a straight line along where it looks, and cannot go
## through the ground or a roof. It may fly over a building but not into its side. First person.
## Tile coordinates: x east, y north, z up; `heading` anticlockwise from east.

const CLEARANCE_M := 0.8
const MAX_ALTITUDE_M := 400.0
const CRUISE_MPS := 12.0
const FAST_MPS := 40.0
const CLIMB_MPS := 8.0
const TURN_RATE := 1.6

var x := 0.0
var y := 0.0
var z := 30.0  # metres above sea level (the tile's ground is its own relief)
var heading := 0.0
var pitch := -0.25
var side_hits := 0  # steps where a building's side stopped it, for the self-test


func place(px: float, py: float, h: float, height_above_ground: float, tile: Tile) -> void:
	x = px
	y = py
	heading = h
	pitch = -0.25
	z = tile.height_at(px, py) + height_above_ground
	side_hits = 0
	sync()


func _floor_at(px: float, py: float, tile: Tile) -> float:
	var roof := tile.roof_at(px, py)
	return maxf(tile.height_at(px, py), roof if roof != Tile.NO_ROOF else -1.0e9) + CLEARANCE_M


## `forward`, `strafe`, `rise` are -1..1 (W, D, E); `turn` -1..1 turns left for positive.
func step(forward: float, strafe: float, rise: float, turn: float, fast: bool, dt: float, tile: Tile) -> void:
	heading += turn * TURN_RATE * dt
	var speed := FAST_MPS if fast else CRUISE_MPS
	var fx := cos(heading)
	var fy := sin(heading)
	# Forward follows the camera's pitch, as it does on a drone with a gimbal, and sideways stays level.
	var along := forward * speed * dt
	var nx := x + (fx * along * cos(pitch)) + fy * strafe * speed * dt
	var ny := y + (fy * along * cos(pitch)) - fx * strafe * speed * dt
	var nz := z + along * sin(pitch) + rise * CLIMB_MPS * dt
	nz = minf(nz, tile.height_at(x, y) + MAX_ALTITUDE_M)
	nx = clampf(nx, 0.0, tile.size_m)
	ny = clampf(ny, 0.0, tile.size_m)
	var roof := tile.roof_at(nx, ny)
	if roof != Tile.NO_ROOF and nz < roof + CLEARANCE_M and z < roof + CLEARANCE_M:
		side_hits += 1  # flying into a wall: stay out, still allow a climb
		nx = x
		ny = y
	x = nx
	y = ny
	z = maxf(nz, _floor_at(x, y, tile))
	sync()


func look(dyaw: float, dpitch: float) -> void:
	heading += dyaw
	pitch = clampf(pitch + dpitch, -1.45, 1.45)


func eye() -> Vector3:
	return Vector3(x, z, -y)


func sync() -> void:
	position = Vector3(x, z, -y)
