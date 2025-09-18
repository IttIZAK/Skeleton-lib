import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:image/image.dart' as img;

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _cameraController;
  late WebSocketChannel _channel;
  bool _isStreaming = false;

  // ---------------- State ----------------
  String? _selectedPose;          // ท่าที่เลือกจาก dropdown
  String _displayPose = "N/A";    // สำหรับแสดงใน Stats Card
  double _confidence = 0.0;
  int _count = 0;
  Map<String, int> _reps = {};
  Map<String, Map<String, double>> _holds = {};

  // ---------------- Config ----------------
  final String serverIp = "10.0.2.2"; // Emulator
  final int serverPort = 8000;
  int _lastSentTime = 0;

  final List<String> _poses = [
    "squat",
    "pushup",
    "plank",
    "situp",
    "forward_lunge",
    "dead_bug",
    "side_plank",
    "russian_twist",
    "lying_leg_raises"
  ];

  @override
  void initState() {
    super.initState();
    _initWebSocket();
    _initCamera();
  }

  @override
  void dispose() {
    _isStreaming = false;
    _cameraController.dispose();
    _channel.sink.close();
    super.dispose();
  }

  // ---------------- WebSocket ----------------
  void _initWebSocket() {
    final uri = Uri.parse('ws://$serverIp:$serverPort/ws/pose');
    print("Connecting to WebSocket: $uri");

    _channel = WebSocketChannel.connect(uri);

    _channel.stream.listen((message) {
      final data = jsonDecode(message);
      if (!mounted) return;

      setState(() {
        // ตรึงชื่อท่าที่เลือก
        _displayPose = _selectedPose ?? "N/A";

        // Update Confidence/Count/Reps/Holds
        _confidence = (data["confidence"] ?? 0).toDouble();
        _count = (data["reps"]?[_selectedPose] ?? 0).toInt();
        _reps = (data["reps"] as Map?)?.map(
              (k, v) => MapEntry(k.toString(), (v as num).toInt()),
            ) ??
            {};
        final holdsData = data["holds"] as Map?;
        if (holdsData != null) {
          _holds = holdsData.map((k, v) {
            final m = v as Map;
            return MapEntry(
              k.toString(),
              {
                "current": (m["current_hold"] ?? 0).toDouble(),
                "best": (m["best_hold"] ?? 0).toDouble(),
              },
            );
          });
        }
      });
    }, onError: (error) {
      print("WebSocket error: $error");
    }, onDone: () {
      print("WebSocket closed");
    });
  }

  // ---------------- Camera ----------------
  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final camera = cameras.firstWhere(
      (cam) => cam.lensDirection == CameraLensDirection.front,
    );

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _cameraController.initialize();
    setState(() {});
    _startImageStream();
  }

  void _startImageStream() {
    _isStreaming = true;
    _cameraController.startImageStream((CameraImage image) async {
      if (!_isStreaming) return;

      final now = DateTime.now().millisecondsSinceEpoch;
      if (now - _lastSentTime < 200) return;
      _lastSentTime = now;

      try {
        if (_channel.closeCode != null) return;

        final jpgBytes = await _convertCameraImageToJpg(image);
        final base64Image = base64Encode(jpgBytes);
        _channel.sink.add(base64Image);
      } catch (e) {
        print("Image conversion/send error: $e");
      }
    });
  }

  Future<Uint8List> _convertCameraImageToJpg(CameraImage image) async {
    final int width = image.width;
    final int height = image.height;
    final img.Image imgImage = img.Image(width: width, height: height);

    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
        final int index = y * width + x;

        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];

        int r = (yp + 1.403 * (vp - 128)).round();
        int g = (yp - 0.344 * (up - 128) - 0.714 * (vp - 128)).round();
        int b = (yp + 1.770 * (up - 128)).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        imgImage.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    return Uint8List.fromList(img.encodeJpg(imgImage));
  }

  // ---------------- Pose Selector ----------------
  Widget _buildPoseSelector() {
    return DropdownButton<String>(
      value: _selectedPose,
      hint: const Text("Select Pose", style: TextStyle(color: Colors.white)),
      dropdownColor: Colors.black87,
      items: _poses.map((pose) {
        return DropdownMenuItem(
          value: pose,
          child: Text(pose, style: const TextStyle(color: Colors.white)),
        );
      }).toList(),
      onChanged: (value) {
        setState(() => _selectedPose = value);
        if (value != null) {
          _channel.sink.add(jsonEncode({"select_pose": value}));
        }
      },
    );
  }

  // ---------------- Build ----------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _cameraController.value.isInitialized
          ? Stack(
              children: [
                CameraPreview(_cameraController),
                Container(
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Colors.black54, Colors.transparent],
                      begin: Alignment.bottomCenter,
                      end: Alignment.topCenter,
                    ),
                  ),
                ),
                Positioned(
                  top: 50,
                  left: 16,
                  right: 16,
                  child: _buildPoseSelector(),
                ),
                Positioned(
                  bottom: 30,
                  left: 16,
                  right: 16,
                  child: _buildStatsCard(),
                ),
              ],
            )
          : const Center(child: CircularProgressIndicator()),
    );
  }

  // ---------------- Stats Card ----------------
  Widget _buildStatsCard() {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      curve: Curves.easeInOut,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white24, width: 1),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.2),
            blurRadius: 12,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Pose Name ตรึงไว้
          Row(
            children: [
              const Icon(Icons.fitness_center, color: Colors.white),
              const SizedBox(width: 8),
              Text(
                _selectedPose ?? "Select Pose",
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 18,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          // Confidence
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            child: Text(
              "Confidence: ${(_confidence * 100).toStringAsFixed(1)}%",
              key: ValueKey(_confidence.toStringAsFixed(1)),
              style: const TextStyle(color: Colors.white),
            ),
          ),
          const SizedBox(height: 4),
          TweenAnimationBuilder<double>(
            tween: Tween(begin: 0, end: _confidence),
            duration: const Duration(milliseconds: 500),
            curve: Curves.easeOutCubic,
            builder: (context, value, _) => LinearProgressIndicator(
              value: value,
              backgroundColor: Colors.white24,
              color: Colors.greenAccent,
              minHeight: 8,
            ),
          ),
          const SizedBox(height: 16),
          // Reps ของท่าที่เลือกเท่านั้น
          if (_selectedPose != null)
            Row(
              children: [
                const Icon(Icons.repeat, color: Colors.white),
                const SizedBox(width: 8),
                Text(
                  "Reps: ${_reps[_selectedPose] ?? 0}",
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          const SizedBox(height: 16),
          // Hold Times ของท่าที่เลือกเท่านั้น
          if (_selectedPose != null && _holds[_selectedPose] != null)
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Hold Times:",
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  "Current: ${_holds[_selectedPose]!["current"]!.toStringAsFixed(1)}s | Best: ${_holds[_selectedPose]!["best"]!.toStringAsFixed(1)}s",
                  style: const TextStyle(color: Colors.white),
                ),
              ],
            ),
        ],
      ),
    );
  }
}