class_name Agents
extends Node3D
## Replays the simulated cars, people and drones (FRK-94). `fraktal scene --agents` writes their
## tracks (`agents.json`, `people.json`, `drones.json`: x, y and heading per agent per sample) beside
## the tile; this plays the loop back, interpolating between samples, and generates nothing. Shapes
## come from `shapes/agent_shapes.json`, which is exported from blender_scene.py, so a car here is the
## rounded car of the Blender scene. Agents do not react to the player and the player does not
## collide with them.

const SHAPES := "res://shapes/agent_shapes.json"
const CORNERS := [Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)]
const FACES := [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7], [3, 2, 1, 0]]
# Display colours (as in Blender's raw view). "body" and "shirt" are white: the instance colour tints them.
const ROLE_COLOURS := {
	"body": Color(1, 1, 1), "shirt": Color(1, 1, 1), "glass": Color(0.05, 0.07, 0.1),
	"tyre": Color(0.03, 0.03, 0.03), "trousers": Color(0.14, 0.15, 0.2), "skin": Color(0.55, 0.38, 0.28),
	"arm": Color(0.75, 0.75, 0.78),
}
const CAR_PAINT := [
	Color8(235, 235, 235), Color8(40, 40, 45), Color8(150, 30, 30), Color8(170, 175, 180), Color8(30, 60, 120),
]
const SHIRTS := [
	Color(0.75, 0.15, 0.15), Color(0.15, 0.3, 0.65), Color(0.85, 0.8, 0.7), Color(0.2, 0.5, 0.25), Color(0.9, 0.6, 0.1),
]

## The simulation runs over the whole world, so most cars spend part of the loop far outside the tile,
## where there is no ground: they are scaled to nothing until they drive back in.
const HIDDEN := Transform3D(Basis(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0)), Vector3(0, -1000, 0))
const EDGE_M := 2.0

var duration_s := 0.0
var stats := {}
var _groups: Array = []  # {"kind", "n", "x", "y", "heading", "z", "multis": [MultiMesh]}
var _step_s := 0.25
var _samples := 0


func load_agents(dir: String, tile: Tile) -> void:
	var shapes: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(SHAPES))
	for entry in [["agents.json", "car"], ["people.json", "person"], ["drones.json", "drone"]]:
		var path: String = dir.path_join(entry[0])
		if not FileAccess.file_exists(path):
			continue
		var d: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(path))
		_add_group(entry[1], d, shapes[entry[1]], tile)
	stats["agent_samples"] = _samples
	stats["agent_loop_s"] = duration_s


func _add_group(kind: String, d: Dictionary, parts: Array, tile: Tile) -> void:
	var n: int = d["ids"].size()
	if n == 0:
		return
	var times: Array = d["t"]
	_samples = times.size()
	_step_s = float(times[1]) - float(times[0]) if times.size() > 1 else 0.25
	duration_s = _step_s * times.size()
	var group := {
		"kind": kind, "n": n, "x": _rows(d["x"]), "y": _rows(d["y"]), "heading": _rows(d["heading"]),
		"z": _rows(d["z"]) if d.has("z") else [], "multis": [],
	}
	# One instanced mesh per tinted set of parts: the paint or shirt takes the instance colour, the rest keeps its own.
	var sets: Array = [["body"], ["glass", "tyre"]] if kind == "car" else [["shirt"], ["trousers", "skin"]] if kind == "person" else [["body", "arm"]]
	for roles in sets:
		var mesh := _mesh(parts, roles)
		if mesh == null:
			continue
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.use_colors = true
		mm.mesh = mesh
		mm.instance_count = n
		var tinted: bool = roles.has("body") and kind == "car" or roles.has("shirt")
		for k in n:
			var c := Color(1, 1, 1)
			if tinted:
				c = CAR_PAINT[(k * 3 + 1) % CAR_PAINT.size()] if kind == "car" else SHIRTS[k % SHIRTS.size()]
			mm.set_instance_color(k, c)
		var node := MultiMeshInstance3D.new()
		node.name = "%s%d" % [kind.capitalize(), group["multis"].size()]
		node.multimesh = mm
		var mat := StandardMaterial3D.new()
		mat.vertex_color_use_as_albedo = true
		mat.vertex_color_is_srgb = true
		mat.roughness = 0.5
		node.material_override = mat
		add_child(node)
		group["multis"].append(mm)
	_groups.append(group)
	stats["agents_" + kind] = n
	update(0.0, tile)


static func _rows(rows: Array) -> Array:
	var out: Array = []
	for r in rows:
		out.append(PackedFloat32Array(r))
	return out


## One mesh from the parts whose role is in `roles`, in the agent's own frame: x forward, y left, z up
## in the sheet, which is (x, z, -y) here. Each box has eight corners (a ring below, a ring above), so
## it can taper, slope and sit to one side. Faces are flat shaded and emitted reversed for Godot.
static func _mesh(parts: Array, roles: Array) -> ArrayMesh:
	var verts := PackedVector3Array()
	var normals := PackedVector3Array()
	var colours := PackedColorArray()
	for part in parts:
		if not roles.has(part["role"]):
			continue
		var b: Array = part["box"]
		var o: Dictionary = part["opt"]
		var shift: float = b[0]
		var half_l: float = b[1]
		var half_w: float = b[2]
		var z0: float = b[3]
		var z1: float = b[4]
		var side: float = o.get("side", 0.0)
		var top_l: float = o.get("top_l", half_l)
		var top_w: float = o.get("top_w", half_w)
		var top_shift: float = o.get("top_shift", 0.0)
		var z1_front: float = o.get("z1_front", z1)
		var pts: Array[Vector3] = []
		for ring in 2:
			for c in CORNERS:
				var hl := half_l if ring == 0 else top_l
				var hw := half_w if ring == 0 else top_w
				var z := z0 if ring == 0 else (z1_front if c.x > 0.0 else z1)
				var lx: float = shift + c.x * hl + (0.0 if ring == 0 else top_shift)
				var ly: float = side + c.y * hw
				pts.append(Vector3(lx, z, -ly))
		var colour: Color = ROLE_COLOURS[part["role"]]
		for f in FACES:
			var a := pts[f[0]]
			var bb := pts[f[1]]
			var cc := pts[f[2]]
			var dd := pts[f[3]]
			var n := (bb - a).cross(cc - a).normalized()
			for tri in [[a, cc, bb], [a, dd, cc]]:
				for v in tri:
					verts.append(v)
					normals.append(n)
					colours.append(colour)
	if verts.is_empty():
		return null
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_COLOR] = colours
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return mesh


## Where agent `k` of `group` is at `t` seconds into the loop: x, y, heading (and height for drones).
func pose(group: Dictionary, k: int, t: float) -> Array:
	var s := fposmod(t, duration_s) / _step_s
	var i := int(s)
	var j := (i + 1) % _samples
	var f := s - i
	var heading := lerp_angle(group["heading"][k][i], group["heading"][k][j], f)
	var z: float = lerpf(group["z"][k][i], group["z"][k][j], f) if not group["z"].is_empty() else 0.0
	return [lerpf(group["x"][k][i], group["x"][k][j], f), lerpf(group["y"][k][i], group["y"][k][j], f), heading, z]


func update(t: float, tile: Tile) -> void:
	for group in _groups:
		for k in group["n"]:
			var p := pose(group, k, t)
			var xf := HIDDEN
			if on_tile(p[0], p[1], tile):
				xf = Transform3D(Basis(Vector3.UP, p[2]), Vector3(p[0], tile.height_at(p[0], p[1]) + p[3], -p[1]))
			for mm in group["multis"]:
				mm.set_instance_transform(k, xf)


func on_tile(x: float, y: float, tile: Tile) -> bool:
	return x >= -EDGE_M and y >= -EDGE_M and x <= tile.size_m + EDGE_M and y <= tile.size_m + EDGE_M


func group_count() -> int:
	return _groups.size()
