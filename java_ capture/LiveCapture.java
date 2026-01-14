import com.zkteco.biometric.FingerprintSensorEx;
import com.zkteco.biometric.FingerprintSensorErrorCode;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Base64;

/**
 * LiveCapture - Continuous fingerprint capture for Python API integration
 * Keeps device open and continuously scans for fingerprints in real-time
 * 
 * Usage: java -cp .;ZKFingerReader.jar LiveCapture
 * 
 * Output Protocol:
 *   - INIT        : Scanner initializing
 *   - READY       : Scanner ready, waiting for finger
 *   - OK:<base64> : Fingerprint captured (base64-encoded BMP image)
 *   - ERROR:<msg> : Fatal error occurred
 * 
 * Features:
 *   - Continuous scanning mode
 *   - Real-time fingerprint capture
 *   - Base64-encoded BMP output
 *   - Graceful shutdown on SIGTERM/SIGINT
 */
public class LiveCapture {
    private static int fpWidth = 0;
    private static int fpHeight = 0;
    private static long mhDevice = 0;
    private static long mhDB = 0;
    private static volatile boolean running = true;
    
    public static void main(String[] args) {
        // Add shutdown hook for cleanup
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.err.println("Shutdown signal received...");
            running = false;
            cleanup();
        }));
        
        try {
            // Initialize SDK
            System.err.println("Initializing ZKTeco SDK...");
            System.out.println("INIT");
            System.out.flush();
            
            int ret = FingerprintSensorEx.Init();
            if (ret != FingerprintSensorErrorCode.ZKFP_ERR_OK) {
                System.out.println("ERROR:SDK Init failed (error code: " + ret + ")");
                System.exit(1);
            }
            
            // Check for devices
            int deviceCount = FingerprintSensorEx.GetDeviceCount();
            if (deviceCount <= 0) {
                System.out.println("ERROR:No fingerprint scanner detected. Please connect your ZKTeco scanner.");
                FingerprintSensorEx.Terminate();
                System.exit(1);
            }
            
            System.err.println("Found " + deviceCount + " scanner(s)");
            
            // Open first device
            mhDevice = FingerprintSensorEx.OpenDevice(0);
            if (mhDevice == 0) {
                System.out.println("ERROR:Failed to open fingerprint scanner");
                FingerprintSensorEx.Terminate();
                System.exit(1);
            }
            
            System.err.println("Scanner opened successfully");
            
            // Initialize fingerprint database
            mhDB = FingerprintSensorEx.DBInit();
            if (mhDB == 0) {
                System.out.println("ERROR:Failed to initialize fingerprint database");
                cleanup();
                System.exit(1);
            }
            
            // Get image dimensions
            byte[] paramValue = new byte[4];
            int[] size = new int[1];
            
            size[0] = 4;
            FingerprintSensorEx.GetParameters(mhDevice, 1, paramValue, size);
            fpWidth = byteArrayToInt(paramValue);
            
            size[0] = 4;
            FingerprintSensorEx.GetParameters(mhDevice, 2, paramValue, size);
            fpHeight = byteArrayToInt(paramValue);
            
            if (fpWidth <= 0 || fpHeight <= 0) {
                System.out.println("ERROR:Invalid image dimensions: " + fpWidth + "x" + fpHeight);
                cleanup();
                System.exit(1);
            }
            
            System.err.println("Image dimensions: " + fpWidth + "x" + fpHeight);
            System.out.println("READY");
            System.out.flush();
            
            // Prepare buffers
            byte[] imgbuf = new byte[fpWidth * fpHeight];
            byte[] template = new byte[2048];
            int[] templateLen = new int[1];
            
            // Main loop - continuously scan for fingerprints
            System.err.println("Entering live capture mode. Waiting for fingerprints...");
            
            long lastCaptureTime = 0;
            int captureCount = 0;
            
            while (running) {
                templateLen[0] = 2048;
                ret = FingerprintSensorEx.AcquireFingerprint(mhDevice, imgbuf, template, templateLen);
                
                if (ret == FingerprintSensorErrorCode.ZKFP_ERR_OK) {
                    // Successfully captured fingerprint
                    captureCount++;
                    long currentTime = System.currentTimeMillis();
                    
                    System.err.println("✓ Fingerprint captured #" + captureCount + " at " + currentTime);
                    
                    try {
                        // Convert to BMP and output as base64
                        byte[] bmpData = convertToBMP(imgbuf, fpWidth, fpHeight);
                        String base64 = Base64.getEncoder().encodeToString(bmpData);
                        System.out.println("OK:" + base64);
                        System.out.flush();
                        
                        // Small delay to avoid capturing the same finger multiple times
                        // Wait for finger to be lifted
                        lastCaptureTime = currentTime;
                        Thread.sleep(1000);
                        
                    } catch (Exception e) {
                        System.err.println("Error processing fingerprint: " + e.getMessage());
                        e.printStackTrace(System.err);
                    }
                    
                } else if (ret == FingerprintSensorErrorCode.ZKFP_ERR_TIMEOUT) {
                    // No finger detected - this is normal, continue waiting
                    // Don't spam logs - only log periodically
                    
                } else if (ret == -8) {
                    // ZKFP_ERR_EXTRACT_FAIL - poor quality
                    // Log occasionally but continue
                    System.err.println("Poor image quality detected (error -8). Waiting for better placement...");
                    
                } else if (ret == -18) {
                    // ZKFP_ERR_NOT_ENOUGH_MEMORY
                    System.err.println("Memory error (code: " + ret + "). Attempting to continue...");
                    
                } else {
                    // Other errors - log but continue trying
                    System.err.println("Capture error code: " + ret + " (continuing...)");
                }
                
                // Small delay between attempts to reduce CPU usage
                try {
                    Thread.sleep(100); // Check every 100ms
                } catch (InterruptedException e) {
                    System.err.println("Interrupted, shutting down...");
                    break;
                }
            }
            
            System.err.println("Exiting live capture mode...");
            cleanup();
            
        } catch (Exception e) {
            System.out.println("ERROR:" + e.getMessage());
            e.printStackTrace(System.err);
            cleanup();
            System.exit(1);
        }
    }
    
    /**
     * Convert raw grayscale image to BMP format (8-bit grayscale)
     * 
     * BMP file format:
     *   - 14 bytes: BMP Header
     *   - 40 bytes: DIB Header (BITMAPINFOHEADER)
     *   - 1024 bytes: Color palette (256 grayscale entries, 4 bytes each)
     *   - Variable: Pixel data (bottom-up, padded to 4-byte boundary)
     */
    private static byte[] convertToBMP(byte[] imageBuf, int width, int height) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        
        // BMP requires width to be multiple of 4 bytes
        int stride = ((width + 3) / 4) * 4;
        int imageSize = stride * height;
        int fileSize = 54 + 1024 + imageSize;
        
        // --- BMP Header (14 bytes) ---
        dos.writeShort(0x424D); // 'BM' signature
        dos.writeInt(Integer.reverseBytes(fileSize)); // File size
        dos.writeShort(0); // Reserved
        dos.writeShort(0); // Reserved
        dos.writeInt(Integer.reverseBytes(54 + 1024)); // Offset to pixel data
        
        // --- DIB Header (40 bytes - BITMAPINFOHEADER) ---
        dos.writeInt(Integer.reverseBytes(40)); // Header size
        dos.writeInt(Integer.reverseBytes(width)); // Width
        dos.writeInt(Integer.reverseBytes(height)); // Height (positive = bottom-up)
        dos.writeShort(Short.reverseBytes((short)1)); // Color planes (must be 1)
        dos.writeShort(Short.reverseBytes((short)8)); // Bits per pixel (8 = grayscale)
        dos.writeInt(0); // Compression (0 = none)
        dos.writeInt(Integer.reverseBytes(imageSize)); // Image size
        dos.writeInt(0); // X pixels per meter (0 = not specified)
        dos.writeInt(0); // Y pixels per meter (0 = not specified)
        dos.writeInt(0); // Colors used (0 = all)
        dos.writeInt(0); // Important colors (0 = all)
        
        // --- Color Palette (256 entries, 4 bytes each = 1024 bytes) ---
        // For grayscale: each entry is (B, G, R, Reserved) with same value
        for (int i = 0; i < 256; i++) {
            dos.writeByte(i); // Blue
            dos.writeByte(i); // Green
            dos.writeByte(i); // Red
            dos.writeByte(0); // Reserved
        }
        
        // --- Pixel Data (bottom-up, padded to stride) ---
        byte[] padding = new byte[stride - width];
        
        // BMP stores pixels bottom-up (last row first)
        for (int y = height - 1; y >= 0; y--) {
            dos.write(imageBuf, y * width, width);
            if (padding.length > 0) {
                dos.write(padding); // Pad each row to 4-byte boundary
            }
        }
        
        dos.flush();
        return baos.toByteArray();
    }
    
    /**
     * Convert 4-byte array to int (little-endian)
     * Used for parsing SDK parameters
     */
    private static int byteArrayToInt(byte[] bytes) {
        int value = bytes[0] & 0xFF;
        value |= ((bytes[1] << 8) & 0xFF00);
        value |= ((bytes[2] << 16) & 0xFF0000);
        value |= ((bytes[3] << 24) & 0xFF000000);
        return value;
    }
    
    /**
     * Cleanup resources
     * Called on shutdown or error
     */
    private static void cleanup() {
        try {
            System.err.println("Cleaning up resources...");
            
            if (mhDB != 0) {
                FingerprintSensorEx.DBFree(mhDB);
                mhDB = 0;
                System.err.println("Database freed");
            }
            
            if (mhDevice != 0) {
                FingerprintSensorEx.CloseDevice(mhDevice);
                mhDevice = 0;
                System.err.println("Device closed");
            }
            
            FingerprintSensorEx.Terminate();
            System.err.println("SDK terminated");
            
        } catch (Exception e) {
            System.err.println("Error during cleanup: " + e.getMessage());
        }
    }
}
