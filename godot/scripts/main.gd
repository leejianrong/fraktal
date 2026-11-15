extends Node3D
## The viewer: loads the tile named by `--scene DIR` and lets you drive a car, walk as a person or
## fly a drone in it. Keys 1, 2 and 3 switch; the new mode starts where the last one was.
## Run: godot --path godot -- --scene DIR   (see godot/README.md). Options after `--`:
##   --assets DIR      the repo's assets directory (default: ../assets next to this project)
##   --mode MODE       start as car (default), person or drone
##   --autodrive       a fixed scripted run of the current mode instead of the keyboard
##   --selftest        no window needed: load, drive 60 s of steps, check, print STATE, exit 0 or 1
##   --bench SECONDS   autodrive for that long, print frame statistics, exit
##   --shot PATH       with --bench, save the last frame as a PNG

const DT := 1.0 / 60.0
const SELFTEST_STEPS := 3600

var args := {}
var tile: Tile
var car: Car
var walker: Walker
var drone: Drone
var mode := "car"
var camera: Camera3D
var hud: Label
var clock := 0.0
var frame_ms: Array[float] = []
var _cam_ready := false


func _ready() -> void:
	args = _parse(OS.get_cmdline_user_args())
	var scene_dir: String = args.get("scene", OS.get_environment("FRAKTAL_SCENE"))
	if scene_dir == "":
		_fail("pass --scene DIR (a directory written by `fraktal scene`)")
		return
	var assets: String = args.get("assets", ProjectSettings.globalize_path("res://").path_join("../assets").simplify_path())
	tile = Tile.new()
	tile.name = "Tile"
	add_child(tile)
	tile.load_tile(scene_dir, assets)
	_log("tile_loaded", tile.stats)
	car = Car.new()
	add_child(car)
	var start := _start(scene_dir)
	car.place(start[0], start[1], start[2], tile)
	walker = Walker.new()
	add_child(walker)
	drone = Drone.new()
	add_child(drone)
	if args.has("selftest"):
		_selftest()
		return
	_environment()
	_camera_and_hud()
	_set_mode(String(args.get("mode", "car")))
	if args.has("bench"):
		args["autodrive"] = true


func _parse(raw: PackedStringArray) -> Dictionary:
	var out := {}
	var i := 0
	while i < raw.size():
		var key := raw[i].trim_prefix("--")
		if i + 1 < raw.size() and not raw[i + 1].begins_with("--"):
			out[key] = raw[i + 1]
			i += 2
		else:
			out[key] = true
			i += 1
	return out


func _fail(msg: String) -> void:
	printerr("error: " + msg)
	get_tree().quit(1)


func _log(event: String, fields: Dictionary) -> void:
	var d := fields.duplicate()
	d["event"] = "godot_" + event
	printerr("fraktal-log " + JSON.stringify(d, "", true))


func _start(scene_dir: String) -> Array:
	var path := scene_dir.path_join("start.json")
	if FileAccess.file_exists(path):
		var s: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(path))
		return [float(s["x"]), float(s["y"]), float(s["heading"])]
	return [tile.size_m / 2.0, tile.size_m / 2.0, 0.0]


func _environment() -> void:
	var sun: Dictionary = tile.scene["sun"]
	var az := deg_to_rad(float(sun["azimuth_deg"]))
	var el := deg_to_rad(float(sun["elevation_deg"]))
	# Azimuth is clockwise from north (tile axes); the light travels away from the sun.
	var toward := Tile.to_engine(Vector3(sin(az) * cos(el), cos(az) * cos(el), sin(el)))
	var light := DirectionalLight3D.new()
	light.shadow_enabled = true
	light.directional_shadow_max_distance = 300.0
	add_child(light)
	light.look_at_from_position(Vector3.ZERO, -toward)
	var sky := ProceduralSkyMaterial.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	env.sky = Sky.new()
	env.sky.sky_material = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.62, 0.72, 0.9).linear_to_srgb()
	env.ambient_light_energy = 0.55
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	var holder := WorldEnvironment.new()
	holder.environment = env
	add_child(holder)


func _camera_and_hud() -> void:
	camera = Camera3D.new()
	camera.far = 3000.0
	camera.fov = 70.0
	add_child(camera)
	camera.make_current()
	var layer := CanvasLayer.new()
	hud = Label.new()
	hud.position = Vector2(12, 8)
	hud.add_theme_color_override("font_color", Color(1, 1, 1))
	hud.add_theme_color_override("font_shadow_color", Color(0, 0, 0))
	hud.add_theme_constant_override("shadow_offset_x", 1)
	hud.add_theme_constant_override("shadow_offset_y", 1)
	layer.add_child(hud)
	add_child(layer)


func _auto(t: float) -> Array:  # throttle, turn: reach 10 m/s straight along the road, then weave
	return [1.0 if car.speed < 10.0 else 0.0, 0.6 * sin(t * 0.5) if t > 5.0 else 0.0]


func _physics_process(_delta: float) -> void:
	if car == null or camera == null:
		return
	var auto := args.has("autodrive")
	var fwd := _axis(KEY_W, KEY_S)
	var side := _axis(KEY_D, KEY_A)  # right is positive for walking and flying
	var turn := _axis(KEY_LEFT, KEY_RIGHT)  # left is positive
	var look_up := _axis(KEY_UP, KEY_DOWN)
	if mode == "car":
		var inp: Array
		if auto:
			inp = _auto(clock)
		else:
			inp = [
				(1.0 if _key(KEY_W) or _key(KEY_UP) else 0.0) - (1.0 if _key(KEY_S) or _key(KEY_DOWN) else 0.0),
				(1.0 if _key(KEY_A) or _key(KEY_LEFT) else 0.0) - (1.0 if _key(KEY_D) or _key(KEY_RIGHT) else 0.0),
			]
		car.step(inp[0], inp[1], _key(KEY_SPACE), DT, tile)
		car.sync(tile)
	elif mode == "person":
		if auto:
			walker.step(1.0, 0.0, 0.3 * sin(clock * 0.4), false, DT, tile)
		else:
			walker.look(0.0, look_up * 1.2 * DT)
			walker.step(fwd, side, turn, _key(KEY_SHIFT), DT, tile)
	else:
		if auto:
			drone.step(1.0, 0.0, 0.4 if clock < 6.0 else 0.0, 0.2 * sin(clock * 0.3), false, DT, tile)
		else:
			drone.look(0.0, look_up * 1.2 * DT)
			drone.step(fwd, side, _axis(KEY_E, KEY_Q), turn, _key(KEY_SHIFT), DT, tile)
	clock += DT
	if Input.is_key_pressed(KEY_R):
		var s := _start(args.get("scene", OS.get_environment("FRAKTAL_SCENE")))
		_place_all(s[0], s[1], s[2])
		_snap_camera()


func _axis(plus: Key, minus: Key) -> float:
	return (1.0 if Input.is_key_pressed(plus) else 0.0) - (1.0 if Input.is_key_pressed(minus) else 0.0)


func _place_all(x: float, y: float, h: float) -> void:
	car.place(x, y, h, tile)
	walker.place(x, y, h, tile)
	drone.place(x, y, h, 30.0, tile)


## The current mode's position, so the next one starts where you are.
func _where() -> Array:
	match mode:
		"person":
			return [walker.x, walker.y, walker.heading]
		"drone":
			return [drone.x, drone.y, drone.heading]
	return [car.x, car.y, car.heading]


func _set_mode(next: String) -> void:
	if not ["car", "person", "drone"].has(next):
		next = "car"
	var at := _where()
	if next != mode or clock == 0.0:
		match next:
			"person":  # beside the car, so you can see where you left it
				walker.place(at[0] - sin(at[2]) * 3.0, at[1] + cos(at[2]) * 3.0, at[2], tile)
			"drone":
				drone.place(at[0], at[1], at[2], 30.0, tile)
			_:
				if tile.roof_at(at[0], at[1]) == Tile.NO_ROOF:
					car.place(at[0], at[1], at[2], tile)
	mode = next
	if mode != "car":
		_cam_ready = false
	_snap_camera()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_1:
				_set_mode("car")
			KEY_2:
				_set_mode("person")
			KEY_3:
				_set_mode("drone")
			KEY_ESCAPE:
				Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseButton and event.pressed and mode != "car":
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	elif event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED and mode != "car":
		var active: Node = walker if mode == "person" else drone
		active.look(-event.relative.x * 0.003, -event.relative.y * 0.003)


func _key(k: Key) -> bool:
	return Input.is_key_pressed(k)


func _snap_camera() -> void:
	_follow(1.0)


func _first_person(at: Vector3, heading: float, pitch: float) -> void:
	var dir := Vector3(cos(heading) * cos(pitch), sin(pitch), -sin(heading) * cos(pitch))
	camera.position = at
	camera.look_at(at + dir)


func _follow(weight: float) -> void:
	if mode == "person":
		_first_person(walker.eye(tile), walker.heading, walker.pitch)
		return
	if mode == "drone":
		_first_person(drone.eye(), drone.heading, drone.pitch)
		return
	var fwd := Vector3(cos(car.heading), 0.0, -sin(car.heading))
	var want := car.position - fwd * 9.0 + Vector3.UP * 3.8
	camera.position = camera.position.lerp(want, weight) if _cam_ready else want
	_cam_ready = true
	camera.look_at(car.position + Vector3.UP * 1.2)


func _process(delta: float) -> void:
	if camera == null:
		return
	_follow(1.0 - exp(-6.0 * delta))
	frame_ms.append(delta * 1000.0)
	match mode:
		"person":
			hud.text = "person   %d fps\nW/S/A/D walk, mouse or arrows look, Shift runs, 1 car, 3 drone, R reset" % Engine.get_frames_per_second()
		"drone":
			hud.text = "drone   %3.0f m up   %d fps\nW/S/A/D fly, E up, Q down, mouse or arrows look, Shift fast, 1 car, 2 person, R reset" % [
				drone.z - tile.height_at(drone.x, drone.y), Engine.get_frames_per_second()]
		_:
			hud.text = "car  %3.0f km/h   walls hit %d   %d fps\nW/S throttle and brake, A/D steer, Space handbrake, 2 person, 3 drone, R reset" % [
				absf(car.speed) * 3.6, car.wall_hits, Engine.get_frames_per_second()]
	if args.has("bench") and clock >= float(args["bench"]):
		_finish_bench()


func _finish_bench() -> void:
	var times := frame_ms.slice(10)  # skip the first frames: shader compilation and first uploads
	times.sort()
	var total := 0.0
	for m in times:
		total += m
	var info := {
		"frames": times.size(),
		"mean_ms": snappedf(total / maxf(times.size(), 1), 0.01),
		"p95_ms": snappedf(times[int(times.size() * 0.95)], 0.01),
		"worst_ms": snappedf(times[-1], 0.01),
		"draw_calls": RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME),
		"primitives": RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME),
		"adapter": RenderingServer.get_video_adapter_name(),
		"driver": RenderingServer.get_current_rendering_driver_name(),
		"sim_seconds": snappedf(clock, 0.1),
	}
	_log("bench", info)
	if args.has("shot"):
		get_viewport().get_texture().get_image().save_png(String(args["shot"]))
	get_tree().quit()


## No window needed. Drives a scripted 60 s, then a head-on run at the nearest building, and checks
## that the car never ends a step inside a footprint or off the tile. STATE is what a second run
## must reproduce exactly.
func _selftest() -> void:
	var ok := true
	var cells := tile.cells + 1
	if tile.nodes.size() != cells * cells or tile.stats.get("building_triangles", 0) == 0 or tile.stats.get("ground_triangles", 0) == 0:
		printerr("FAIL: the tile did not load (heights %d, stats %s)" % [tile.nodes.size(), tile.stats])
		ok = false
	var path := 0.0
	var inside := 0
	var px := car.x
	var py := car.y
	for i in SELFTEST_STEPS:
		var a := _auto(i * DT)
		car.step(a[0], a[1], false, DT, tile)
		path += Vector2(car.x - px, car.y - py).length()
		px = car.x
		py = car.y
		if tile.roof_at(car.x, car.y) != Tile.NO_ROOF:
			inside += 1
	car.sync(tile)
	var ground := tile.height_at(car.x, car.y)
	print("STATE %.4f %.4f %.4f %.4f %d" % [car.x, car.y, car.heading, car.speed, car.wall_hits])
	if path < 5.0:
		printerr("FAIL: the scripted drive only covered %.1f m" % path)
		ok = false
	if inside > 0:
		printerr("FAIL: the car was inside a building in %d steps" % inside)
		ok = false
	if absf(car.position.y - ground) > 2.0:
		printerr("FAIL: the car is %.2f m from the ground" % absf(car.position.y - ground))
		ok = false
	# Head-on at the first building we can approach from a free spot 30 m away.
	var rammed := false
	for b in tile.scene["buildings"]:
		var centre := Vector2.ZERO
		for p in b["ring"]:
			centre += Vector2(p[0], p[1])
		centre /= b["ring"].size()
		for k in 8:
			var ang := k * PI / 4.0
			var from := centre + Vector2(cos(ang), sin(ang)) * 30.0
			if from.x < 5 or from.y < 5 or from.x > tile.size_m - 5 or from.y > tile.size_m - 5:
				continue
			if tile.roof_at(from.x, from.y) != Tile.NO_ROOF:
				continue
			car.place(from.x, from.y, atan2(centre.y - from.y, centre.x - from.x) , tile)
			var in_steps := 0
			for i in 900:
				car.step(1.0, 0.0, false, DT, tile)
				if tile.roof_at(car.x, car.y) != Tile.NO_ROOF:
					in_steps += 1
			if car.wall_hits > 0:
				rammed = true
				if in_steps > 0:
					printerr("FAIL: the car drove into a building")
					ok = false
				break
		if rammed:
			break
	print("RAMMED %s" % rammed)
	if not rammed:
		printerr("FAIL: no building was reached head-on")
		ok = false
	ok = _selftest_person() and ok
	ok = _selftest_drone() and ok
	_log("selftest", {"ok": ok, "path_m": snappedf(path, 0.1)})
	get_tree().quit(0 if ok else 1)


## The nearest building we can approach from a free spot `dist` metres away, facing it.
func _approach(dist: float) -> Array:
	for b in tile.scene["buildings"]:
		var centre := Vector2.ZERO
		for p in b["ring"]:
			centre += Vector2(p[0], p[1])
		centre /= b["ring"].size()
		for k in 8:
			var ang := k * PI / 4.0
			var from := centre + Vector2(cos(ang), sin(ang)) * dist
			if from.x < 5 or from.y < 5 or from.x > tile.size_m - 5 or from.y > tile.size_m - 5:
				continue
			if tile.person_blocked(from.x, from.y, Walker.RADIUS_M):
				continue
			return [from.x, from.y, atan2(centre.y - from.y, centre.x - from.x)]
	return []


func _selftest_person() -> bool:
	var ok := true
	var start := _start(args.get("scene", OS.get_environment("FRAKTAL_SCENE")))
	walker.place(start[0], start[1], start[2], tile)
	var path := 0.0
	var bad := 0
	var px := walker.x
	var py := walker.y
	for i in 3000:
		walker.step(1.0, 0.0, 0.3 * sin(i * DT * 0.4), i % 600 > 300, DT, tile)
		path += Vector2(walker.x - px, walker.y - py).length()
		px = walker.x
		py = walker.y
		if tile.person_blocked(walker.x, walker.y, Walker.RADIUS_M):
			bad += 1
	print("WALK %.4f %.4f %.4f" % [walker.x, walker.y, walker.heading])
	if path < 10.0 or bad > 0:
		printerr("FAIL: the walk covered %.1f m and was inside something in %d steps" % [path, bad])
		ok = false
	if absf(walker.eye(tile).y - tile.height_at(walker.x, walker.y) - Walker.EYE_M) > 0.01:
		printerr("FAIL: the eye is not %.1f m above the ground" % Walker.EYE_M)
		ok = false
	var from := _approach(25.0)
	if from.is_empty():
		printerr("FAIL: no building to walk into")
		return false
	walker.place(from[0], from[1], from[2], tile)
	var inside := 0
	for i in 900:
		walker.step(1.0, 0.0, 0.0, false, DT, tile)
		if tile.roof_at(walker.x, walker.y) != Tile.NO_ROOF:
			inside += 1
	print("WALKED_INTO blocked=%d inside=%d" % [walker.blocked_steps, inside])
	if walker.blocked_steps == 0 or inside > 0:
		printerr("FAIL: walking at a building should stop at it (blocked %d, inside %d)" % [walker.blocked_steps, inside])
		ok = false
	return ok


func _selftest_drone() -> bool:
	var ok := true
	var start := _start(args.get("scene", OS.get_environment("FRAKTAL_SCENE")))
	drone.place(start[0], start[1], start[2], 6.0, tile)
	drone.pitch = -1.2  # a steep dive, which the ground must stop
	var below := 0
	for i in 900:
		drone.step(1.0, 0.0, 0.0, 0.0, false, DT, tile)
		if drone.z < tile.height_at(drone.x, drone.y) + Drone.CLEARANCE_M - 0.001:
			below += 1
	print("DIVE %.3f above=%.3f" % [drone.z, drone.z - tile.height_at(drone.x, drone.y)])
	if below > 0:
		printerr("FAIL: the drone went into the ground in %d steps" % below)
		ok = false
	var from := _approach(30.0)
	if from.is_empty():
		printerr("FAIL: no building to fly at")
		return false
	drone.place(from[0], from[1], from[2], 2.0, tile)
	drone.pitch = 0.0
	var through := 0
	for i in 900:
		drone.step(1.0, 0.0, 0.0, 0.0, false, DT, tile)
		var roof := tile.roof_at(drone.x, drone.y)
		if roof != Tile.NO_ROOF and drone.z < roof + Drone.CLEARANCE_M - 0.001:
			through += 1
	print("FLEW_AT side_hits=%d through=%d" % [drone.side_hits, through])
	if drone.side_hits == 0 or through > 0:
		printerr("FAIL: a drone flying at a wall at head height should stop at it (hits %d, through %d)" % [drone.side_hits, through])
		ok = false
	drone.place(from[0], from[1], from[2], 2.0, tile)  # the same approach, climbing: it goes over
	drone.pitch = 0.0
	var crossed := false
	for i in 1800:
		drone.step(1.0, 0.0, 1.0, 0.0, false, DT, tile)
		if tile.roof_at(drone.x, drone.y) != Tile.NO_ROOF:
			crossed = true
	print("OVER crossed=%s z=%.2f" % [crossed, drone.z])
	if not crossed:
		printerr("FAIL: a climbing drone should fly over the building")
		ok = false
	return ok
