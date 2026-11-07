class_name Car
extends Node3D
## The player's car: the browser demo's kinematic bicycle model (src/fraktal/demo_template.html),
## stepped by hand at a fixed rate so a scripted drive is reproducible. Godot's physics engine is not
## used. Tile coordinates: x east, y north, heading anticlockwise from east.

const HALF_LENGTH := 2.25
const HALF_WIDTH := 0.95
const CORNERS := [Vector2(2.2, 0.9), Vector2(2.2, -0.9), Vector2(-2.2, 0.9), Vector2(-2.2, -0.9), Vector2.ZERO]

var x := 0.0
var y := 0.0
var heading := 0.0
var speed := 0.0  # m/s along the heading, negative in reverse
var steer := 0.0  # front wheel angle, radians
var wall_hits := 0


func _init() -> void:
	name = "Car"
	_box(Vector3(4.5, 0.75, 1.9), Vector3(0, 0.625, 0), Color(0.91, 0.27, 0.24))
	_box(Vector3(2.4, 0.5, 1.7), Vector3(-0.2, 1.25, 0), Color(0.05, 0.07, 0.1))


func _box(size: Vector3, at: Vector3, colour: Color) -> void:
	var mesh := BoxMesh.new()
	mesh.size = size
	var mat := StandardMaterial3D.new()
	mat.albedo_color = colour
	mat.roughness = 0.35
	var node := MeshInstance3D.new()
	node.mesh = mesh
	node.material_override = mat
	node.position = at
	add_child(node)


func place(px: float, py: float, h: float, tile: Tile) -> void:
	x = px
	y = py
	heading = h
	speed = 0.0
	steer = 0.0
	wall_hits = 0
	sync(tile)


## One step. `throttle` is -1..1 (W is 1, S is -1), `turn` -1..1 (A, left, is 1).
func step(throttle: float, turn: float, handbrake: bool, dt: float, tile: Tile) -> void:
	if throttle > 0.0:
		speed += 4.0 * dt
	elif throttle < 0.0:
		speed += (-8.0 if speed > 0.3 else -3.0) * dt
	else:
		speed -= signf(speed) * minf(absf(speed), 0.6 * dt)
	if handbrake:
		speed -= signf(speed) * minf(absf(speed), 9.0 * dt)
	speed = clampf(speed, -4.0, 22.0)
	var want := turn * 0.55 / (1.0 + absf(speed) / 10.0)
	steer += (want - steer) * minf(1.0, 8.0 * dt)
	var nh := heading + speed / 2.7 * tan(steer) * dt
	var nx := x + cos(nh) * speed * dt
	var ny := y + sin(nh) * speed * dt
	var hit := false
	for c in CORNERS:
		var px: float = nx + cos(nh) * c.x - sin(nh) * c.y
		var py: float = ny + sin(nh) * c.x + cos(nh) * c.y
		if tile.roof_at(px, py) != Tile.NO_ROOF:
			hit = true
			break
	if hit:
		if absf(speed) > 2.0:
			wall_hits += 1
		speed *= -0.2
	else:
		x = nx
		y = ny
		heading = nh
	x = clampf(x, 0.0, tile.size_m)
	y = clampf(y, 0.0, tile.size_m)


## Put the body on the ground: height from the four wheel points, pitch and roll from their slope.
func sync(tile: Tile) -> void:
	var fx := cos(heading)
	var fy := sin(heading)
	var lx := -fy
	var ly := fx
	var zf := tile.height_at(x + fx * 2.2, y + fy * 2.2)
	var zb := tile.height_at(x - fx * 2.2, y - fy * 2.2)
	var zl := tile.height_at(x + lx * HALF_WIDTH, y + ly * HALF_WIDTH)
	var zr := tile.height_at(x - lx * HALF_WIDTH, y - ly * HALF_WIDTH)
	var pitch := atan2(zf - zb, 4.4)
	var roll := atan2(zl - zr, 2.0 * HALF_WIDTH)
	var basis := Basis(Vector3.UP, heading) * Basis(Vector3(0, 0, 1), pitch) * Basis(Vector3(1, 0, 0), roll)
	transform = Transform3D(basis, Vector3(x, (zf + zb + zl + zr) / 4.0, -y))
