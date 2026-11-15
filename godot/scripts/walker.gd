class_name Walker
extends Node3D
## A person on foot (FRK-92): eye at 1.6 m above the ground, sliding along buildings, compound walls,
## trunks and poles. First person, so there is no body. Tile coordinates: x east, y north, `heading`
## anticlockwise from east. Stepped by hand at a fixed rate so a scripted walk is reproducible.

const EYE_M := 1.6
const RADIUS_M := 0.3
const WALK_MPS := 2.0
const RUN_MPS := 5.0
const TURN_RATE := 1.8  # radians per second at full arrow key

var x := 0.0
var y := 0.0
var heading := 0.0
var pitch := 0.0  # up is positive
var blocked_steps := 0  # steps where a wall stopped a move on either axis, for the self-test


func place(px: float, py: float, h: float, tile: Tile) -> void:
	x = px
	y = py
	heading = h
	pitch = 0.0
	blocked_steps = 0
	# A start inside something steps out the nearest free spot along the heading.
	var tries := 0
	while tile.person_blocked(x, y, RADIUS_M) and tries < 60:
		x += cos(heading) * 0.5
		y += sin(heading) * 0.5
		tries += 1
	sync(tile)


## One step. `forward` and `strafe` are -1..1 (W is forward, D is strafe right); `turn` -1..1 turns
## left for positive; `run` is shift.
func step(forward: float, strafe: float, turn: float, run: bool, dt: float, tile: Tile) -> void:
	heading += turn * TURN_RATE * dt
	var speed := RUN_MPS if run else WALK_MPS
	var fx := cos(heading)
	var fy := sin(heading)
	var dx := (fx * forward + fy * strafe) * speed * dt
	var dy := (fy * forward - fx * strafe) * speed * dt
	var stopped := false
	# Axes separately, so a wall is slid along instead of stopping the walk.
	if dx != 0.0 or dy != 0.0:
		if tile.person_blocked(x + dx, y, RADIUS_M):
			stopped = true
		else:
			x += dx
		if tile.person_blocked(x, y + dy, RADIUS_M):
			stopped = true
		else:
			y += dy
		if stopped:
			blocked_steps += 1
	sync(tile)


func look(dyaw: float, dpitch: float) -> void:
	heading += dyaw
	pitch = clampf(pitch + dpitch, -1.45, 1.45)


func eye(tile: Tile) -> Vector3:
	return Vector3(x, tile.height_at(x, y) + EYE_M, -y)


func sync(tile: Tile) -> void:
	position = Vector3(x, tile.height_at(x, y), -y)
