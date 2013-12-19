
public class Vector {

	/**
	 * Vector dot product
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static double dot(double[] v1, double[] v2) {
		// Force an IndexOutOfBoundsException if lengths aren't equal
		int length = Math.max(v1.length, v2.length);
		double dot = 0D;
		for(int i=0; i < length; i++) {
			dot += v1[i] * v2[i];
		}
		return dot;
	}
	
	/**
	 * Returns a version of the given vector scaled by alpha
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static double[] scaled(double[] v1, double alpha) {
		// Force an IndexOutOfBoundsException if lengths aren't equal
		double[] temp = new double[v1.length];
		for(int i=0; i < v1.length; i++) {
			temp[i] = alpha*v1[i];
		}
		return temp;
	}
	
	/**
	 * Scales the given vector by alpha in-place
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static void scale(double[] v1, double alpha) {
		// Force an IndexOutOfBoundsException if lengths aren't equal
		for(int i=0; i < v1.length; i++) {
			v1[i] = alpha*v1[i];
		}
	}
	
	/**
	 * Returns the sum of two vectors
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static double[] add(double[] v1, double[] v2) {
		// Force an IndexOutOfBoundsException if lengths aren't equal
		int length = Math.max(v1.length, v2.length);
		double[] diff = new double[length];
		for(int i=0; i < length; i++) {
			diff[i] = v1[i] + v2[i];
		}
		return diff;
	}
	
	
	
	/**
	 * Returns the difference of two vectors
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static double[] sub(double[] v1, double[] v2) {
		// Force an IndexOutOfBoundsException if lengths aren't equal
		int length = Math.max(v1.length, v2.length);
		double[] diff = new double[length];
		for(int i=0; i < length; i++) {
			diff[i] = v1[i] - v2[i];
		}
		return diff;
	}

	public static double max(double[] xs) {
		// IndexOutOfBounds will be intentional if it occurs
		double max = xs[0];
		for(int i=0; i < xs.length; i++) {
			max = Math.max(max, xs[i]);
		}
		return max;
	}
	
	/**
	 * Adds v1 to v2 in place. v2 is the destination array.
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static void addTo(double[] v1, double[] v2) {
		for(int i=0; i < v2.length; i++) {
			v2[i] = v1[i] + v2[i];
		}
	}
	
	/**
	 * Adds v1 to v2 in place. v2 is the destination array.
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static void addTo(double val, double[] v2) {
		for(int i=0; i < v2.length; i++) {
			v2[i] = val + v2[i];
		}
	}
	
	/**
	 * Subtracts val from each item in v2, in place in v2.
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static void subFrom(double val, double[] v2) {
		for(int i=0; i < v2.length; i++) {
			v2[i] = val + v2[i];
		}
	}

	/**
	 * Subtracts v1 from items in v2, in place in v2.
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static void subFrom(double[] v1, double[] v2) {
		for(int i=0; i < v2.length; i++) {
			v2[i] = v2[i] - v1[i];
		}
	}
	
}
