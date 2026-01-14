import com.zkteco.biometric.FingerprintSensorEx;
import com.zkteco.biometric.FingerprintSensorErrorCode;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Base64;

/**
 * CaptureOnce - Single fingerprint capture helper for Python API integration
 * Based on ZKFinger SDK Demo (ZKFPDemo.java)
 * 
 * Usage: java -cp .;ZKFingerReader.jar CaptureOnce
 * Output: OK:<base64_png> on success, ERROR:<message> on failure
 */
public class CaptureOnce {
    private static int fpWidth = 0;
    private static int fpHeight = 0;
    private static long mhDevice = 0;
    private static long mhDB = 0;
    
    public static void main(String[] args) {
        try {
            // Initialize SDK
            int ret = FingerprintSensorEx.Init();
            if (ret != FingerprintSensorErrorCode.ZKFP_ERR_OK) {
                System.out.println("ERROR:SDK Init failed");
                System.exit(1);
            }
            
            // Check for devices
            int deviceCount = FingerprintSensorEx.GetDeviceCount();
            if (deviceCount <= 0) {
                System.out.println("ERROR:No fingerprint scanner detected. Please connect your ZKTeco scanner.");
                FingerprintSensorEx.Terminate();
                System.exit(1);
            }
            
            // Open first device
            mhDevice = FingerprintSensorEx.OpenDevice(0);
            if (mhDevice == 0) {
                System.out.println("ERROR:Failed to open fingerprint scanner");
                FingerprintSensorEx.Terminate();
                System.exit(1);
            }
            
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
            
            // Prepare buffers
            byte[] imgbuf = new byte[fpWidth * fpHeight];
            byte[] template = new byte[2048];
            int[] templateLen = new int[1];
            templateLen[0] = 2048;
            
            // Wait for fingerprint (blocking call with timeout)
            System.err.println("Waiting for fingerprint... Place finger on scanner.");
            
            // Try to capture with timeout (retry up to 30 seconds)
            // Error codes:
            // 0 = ZKFP_ERR_OK (success)
            // -5 = ZKFP_ERR_TIMEOUT (no finger - keep trying)
            // -8 = ZKFP_ERR_EXTRACT_FAIL (bad image quality - keep trying)
            // -18 = ZKFP_ERR_NOT_ENOUGH_MEMORY
            boolean captured = false;
            int maxAttempts = 60; // 60 * 500ms = 30 seconds
            int attempt = 0;
            int extractFailCount = 0;
            int maxExtractFails = 10; // Increased from 3 to 10 attempts
            
            System.err.println("Waiting for finger... (will retry for 30 seconds)");
            
            while (!captured && attempt < maxAttempts) {
                ret = FingerprintSensorEx.AcquireFingerprint(mhDevice, imgbuf, template, templateLen);
                
                if (ret == FingerprintSensorErrorCode.ZKFP_ERR_OK) {
                    captured = true;
                    System.err.println("Fingerprint captured successfully!");
                } else if (ret == FingerprintSensorErrorCode.ZKFP_ERR_TIMEOUT) {
                    // No finger detected - continue waiting
                    attempt++;
                    if (attempt % 10 == 0) {
                        System.err.println("Still waiting for finger... (" + (attempt / 2) + "s)");
                    }
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        break;
                    }
                } else if (ret == -8) {
                    // ZKFP_ERR_EXTRACT_FAIL - bad quality, retry
                    extractFailCount++;
                    attempt++;
                    if (extractFailCount <= maxExtractFails) {
                        System.err.println("Poor image quality (attempt " + extractFailCount + "/" + maxExtractFails + "). Press MUCH FIRMER and hold COMPLETELY STILL...");
                    } else {
                        System.out.println("ERROR:Failed to capture after " + extractFailCount + " attempts. Image quality too poor. Press FIRMER!");
                        cleanup();
                        System.exit(1);
                    }
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        break;
                    }
                } else {
                    // Other error - fatal
                    System.out.println("ERROR:Capture failed with error code: " + ret);
                    cleanup();
                    System.exit(1);
                }
            }
            
            if (!captured) {
                System.out.println("ERROR:Timeout - No fingerprint detected after 30 seconds");
                cleanup();
                System.exit(1);
            }
            
            // Convert image buffer to BMP format (grayscale 8-bit)
            byte[] bmpData = convertToBMP(imgbuf, fpWidth, fpHeight);
            
            // Encode to base64 and output
            String base64 = Base64.getEncoder().encodeToString(bmpData);
            System.out.println("OK:" + base64);
            
            // Cleanup
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
     */
    private static byte[] convertToBMP(byte[] imageBuf, int width, int height) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        
        // BMP requires width to be multiple of 4 bytes
        int stride = ((width + 3) / 4) * 4;
        int imageSize = stride * height;
        
        // BMP Header (14 bytes)
        dos.writeShort(0x424D); // 'BM'
        dos.writeInt(Integer.reverseBytes(54 + 1024 + imageSize)); // File size
        dos.writeShort(0); // Reserved
        dos.writeShort(0); // Reserved
        dos.writeInt(Integer.reverseBytes(54 + 1024)); // Offset to pixel data
        
        // DIB Header (40 bytes)
        dos.writeInt(Integer.reverseBytes(40)); // Header size
        dos.writeInt(Integer.reverseBytes(width)); // Width
        dos.writeInt(Integer.reverseBytes(height)); // Height
        dos.writeShort(Short.reverseBytes((short)1)); // Planes
        dos.writeShort(Short.reverseBytes((short)8)); // Bits per pixel
        dos.writeInt(0); // Compression (none)
        dos.writeInt(Integer.reverseBytes(imageSize)); // Image size
        dos.writeInt(0); // X pixels per meter
        dos.writeInt(0); // Y pixels per meter
        dos.writeInt(0); // Colors used (0 = all)
        dos.writeInt(0); // Important colors (0 = all)
        
        // Color palette (256 grayscale entries)
        for (int i = 0; i < 256; i++) {
            dos.writeByte(i); // Blue
            dos.writeByte(i); // Green
            dos.writeByte(i); // Red
            dos.writeByte(0); // Reserved
        }
        
        // Pixel data (bottom-up, padded to stride)
        byte[] padding = new byte[stride - width];
        for (int y = height - 1; y >= 0; y--) {
            dos.write(imageBuf, y * width, width);
            if (padding.length > 0) {
                dos.write(padding);
            }
        }
        
        dos.flush();
        return baos.toByteArray();
    }
    
    /**
     * Convert 4-byte array to int (little-endian)
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
     */
    private static void cleanup() {
        try {
            if (mhDB != 0) {
                FingerprintSensorEx.DBFree(mhDB);
                mhDB = 0;
            }
            if (mhDevice != 0) {
                FingerprintSensorEx.CloseDevice(mhDevice);
                mhDevice = 0;
            }
            FingerprintSensorEx.Terminate();
        } catch (Exception e) {
            // Ignore cleanup errors
        }
    }
}

