package com.spatial.restaurantfinder.spatial;

import com.spatial.restaurantfinder.model.Restaurant;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.function.Predicate;

/**
 * In-memory 2-dimensional KD-Tree over restaurant coordinates, used for k-nearest-neighbour (k-NN) search.
 * Nodes at even depth split on latitude, nodes at odd depth split on longitude.
 *
 * WHY A KD-TREE
 * Finding the k closest restaurants by measuring the distance to every one of the n restaurants costs
 * O(n log k) per query, and that work is repeated for every query. A KD-Tree is built once. Each query then
 * walks down the tree and skips whole groups of restaurants that are provably too far away, so far fewer
 * than n restaurants are examined. The idea is the same as looking up a word in a dictionary: one comparison
 * rules out half of the remaining candidates.
 *
 * HOW DISTANCE IS MEASURED
 * Ranking uses the squared distance dLat² + (dLng · cos(queryLat))² (an equirectangular projection around
 * the query point). Squaring keeps the same ordering without a square root per node, and the cos(lat) factor
 * makes a degree of longitude comparable to a degree of latitude. Each result also carries the exact
 * Haversine distance in kilometres. Queries do not wrap across the ±180° meridian. The class is not
 * thread-safe: build once, then query from a single thread.
 *
 * TIME COMPLEXITY ANALYSIS
 *
 * Notation. n = number of restaurants in the tree, k = number of neighbours requested, log = base-2
 * logarithm (how many times n can be halved before reaching 1, e.g. log 1,000,000 ≈ 20).
 * O(f) means "grows at most as fast as f", Ω(f) means "grows at least as fast as f" and Θ(f) means both,
 * i.e. exactly that growth rate.
 *
 * Summary
 *   build()                      time Θ(n log² n)                  space Θ(n)
 *   findKNearest(), worst case   time O(n log k)                   space O(k + log n)
 *   findKNearest(), expected     time O((log n + k) · log k)       space O(k + log n)
 *
 * 1. The tree is balanced, so its height is only about log n.
 * Each node stores the median point of its range and hands half of the remaining points to each child.
 * Halving repeatedly means that after d levels a range holds at most n / 2^d points, and ranges run out of
 * points once d exceeds log n. The height is therefore ⌈log₂(n + 1)⌉ = O(log n). For example, 40
 * restaurants need only 6 levels, and 1,000,000 restaurants need only 20.
 *
 * 2. Build: Θ(n log² n).
 * Creating the node for a range means sorting that range on the current axis (to find the median) and then
 * building the two halves in the same way. Sorting m items costs Θ(m log m) (Arrays.sort is a merge sort).
 * Counting the cost level by level:
 *   - Level 0 sorts all n points: about n log n work.
 *   - Level 1 sorts two halves of about n/2 points each: again about n log n work in total.
 *   - Every level sorts each remaining point once, so every level costs at most n log n.
 *   - There are about log n levels.
 * Total ≈ (log n levels) × (n log n per level) = n log² n, so the build is O(n log² n). The bound is tight
 * (Ω(n log² n) as well): in the upper half of the levels, at least n/2 points are still being sorted in
 * ranges of at least √n elements, which costs Ω(n log n) per level. The same result follows from the
 * recurrence T(n) = 2·T(n/2) + Θ(n log n) and the Master Theorem. A linear-time median selection in place
 * of the sort would reduce the build to O(n log n); sorting is simpler, and n is small here (Google returns
 * at most 40 places per search).
 *
 * 3. Worst-case query: O(n log k).
 * Each node is visited at most once, so at most n nodes are examined. A visit does a constant amount of
 * arithmetic, plus an O(log k) heap operation when the restaurant enters the current list of k best.
 * Sub-trees can only be skipped after k candidates have been found. If the active filter accepts fewer than
 * k restaurants (for example a rare cuisine), k candidates are never found, nothing is skipped and every
 * node is visited: n visits × O(log k) = O(n log k).
 *
 * 4. Why skipping a sub-tree is safe.
 * Every point on the far side of a splitting plane is at least as far from the query as the plane itself
 * (using the same distance scaling as the ranking). If the plane is not closer than the current k-th best
 * result, no point behind it can improve the result, so the whole sub-tree is skipped. The answer is
 * therefore exact, not approximate.
 *
 * 5. Expected query: O((log n + k) · log k).
 * The search has two parts. First it walks down about log n levels to reach the region around the query
 * point, always taking the nearer side first. Then it checks the neighbouring cells that could still hold
 * something closer. The k nearest restaurants lie inside a small circle around the query that contains about
 * k points; a balanced KD-tree cell holds a constant number of points, so only about k cells intersect
 * that circle. In total about log n + k nodes are visited, and each restaurant that enters the best-k list
 * costs O(log k) for the heap. Example: with 1,000,000 restaurants and k = 10, on the order of 30 nodes are
 * examined (times a small constant) instead of 1,000,000. This agrees with the O(log n) expected visits for
 * fixed k shown by Friedman, Bentley and Finkel (1977). It assumes the points are spread out in a normal
 * way (not adversarially placed) and that k is much smaller than n. The line "visited X of N nodes"
 * printed after each query shows where a particular query landed between the best and worst case.
 *
 * 6. Space.
 * The tree holds n nodes with a constant number of fields each (the restaurant objects are shared, not
 * copied): Θ(n). A query adds a heap of at most k entries, a result list of k entries and a recursion stack
 * as deep as the tree, which is O(log n) because the tree is balanced: O(k + log n).
 */
public final class KDTree {

    private static final int AXIS_LAT = 0;

    /**
     * Comparators used to sort a sub-range on the current axis during the build. The second key
     * (the other coordinate) is a tie-breaker so that sorting is deterministic when two restaurants
     * share the same latitude (or longitude).
     */
    private static final Comparator<Restaurant> BY_LAT =
            Comparator.comparingDouble(Restaurant::lat).thenComparingDouble(Restaurant::lng);
    private static final Comparator<Restaurant> BY_LNG =
            Comparator.comparingDouble(Restaurant::lng).thenComparingDouble(Restaurant::lat);

    /**
     * A k-NN result entry.
     *
     * @param restaurant  the matched restaurant
     * @param distanceSq  squared projected coordinate distance (degrees²) used for ranking
     * @param haversineKm true great-circle distance from the query point in kilometres
     */
    public record RestaurantDistance(Restaurant restaurant, double distanceSq, double haversineKm) {
        /** Time: O(1): a single null check. */
        public RestaurantDistance {
            Objects.requireNonNull(restaurant, "restaurant must not be null");
        }

        /** @return true great-circle distance in metres. Time: O(1): one multiplication. */
        public double haversineMeters() {
            return haversineKm * 1000.0;
        }
    }

    /** Root of the tree; null when the tree is empty. */
    private KDNode root;
    /** Number of restaurants indexed, cached so size is O(1) instead of an O(n) walk. */
    private int size;
    /** Diagnostic counter: how many nodes the last query touched (shows how well pruning worked). */
    private int lastVisitedNodes;

    /**
     * Builds a balanced tree from the given restaurants, replacing any previous contents. The median of each
     * sub-range (sorted on the current axis) becomes a node and the two halves become its children.
     * Time Θ(n log² n): there are about log n levels and each level sorts about n points in total
     * (n log n). Space Θ(n): one node per restaurant. The class comment gives the full derivation.
     */
    public void build(List<Restaurant> restaurants) {
        Objects.requireNonNull(restaurants, "restaurants must not be null");
        // Copy into an array (O(n)) so sub-ranges can be sorted in place without touching the caller's list,
        // and drop any null entries so they can never reach a comparator.
        Restaurant[] points = restaurants.stream()
                .filter(Objects::nonNull)
                .toArray(Restaurant[]::new);
        this.size = points.length;
        this.root = buildRecursive(points, 0, points.length, 0);
    }

    /**
     * Recursively builds the sub-tree for points[fromInclusive, toExclusive).
     *
     * Recurrence: T(m) = 2·T(m/2) + O(m log m) → T(n) = O(n log² n).
     *
     * @return the root of the sub-tree, or null when the range is empty (base case)
     */
    private KDNode buildRecursive(Restaurant[] points, int fromInclusive, int toExclusive, int depth) {
        // Base case: empty range → no node.
        if (fromInclusive >= toExclusive) {
            return null;
        }
        // Pick the axis for this level (lat on even depths, lng on odd) and sort only this sub-range.
        // Arrays.sort on objects is a stable TimSort: O(m log m) for a range of m points.
        Comparator<Restaurant> comparator = (depth % 2 == AXIS_LAT) ? BY_LAT : BY_LNG;
        Arrays.sort(points, fromInclusive, toExclusive, comparator);

        // The middle element is the median; everything before it goes left, everything after goes right.
        // Written as from + (to - from) / 2 instead of (from + to) / 2 to avoid int overflow.
        int median = fromInclusive + (toExclusive - fromInclusive) / 2;
        KDNode node = new KDNode(points[median], depth);
        node.setLeft(buildRecursive(points, fromInclusive, median, depth + 1));
        node.setRight(buildRecursive(points, median + 1, toExclusive, depth + 1));
        return node;
    }

    /**
     * Finds the k restaurants nearest to the target using branch-and-bound search.
     * Same as findKNearest with no filter.
     *
     * @return up to k results sorted by ascending distance; empty when the tree is empty or k ≤ 0
     */
    public List<RestaurantDistance> findKNearest(double targetLat, double targetLng, int k) {
        return findKNearest(targetLat, targetLng, k, null);
    }

    /**
     * Finds the k restaurants nearest to the target that satisfy the given filter (null means no filter),
     * using branch-and-bound search. A max-heap holds the best k found so far, the nearer side of each
     * splitting plane is searched first, and the farther side is skipped when the plane is already farther
     * than the current k-th best. The filter is applied during the search, so k matching results are
     * returned rather than "k nearest, minus those rejected".
     *
     * Time: O(n log k) worst case, because every node may be visited and each can cost an O(log k)
     * heap operation; O((log n + k) · log k) expected, because only about log n + k nodes are visited.
     * Space: O(k + log n) for the heap of k entries plus recursion as deep as the tree. The class
     * comment contains the proofs.
     *
     * @return up to k results sorted by ascending distance; empty when the tree is empty or k <= 0
     */
    public List<RestaurantDistance> findKNearest(double targetLat, double targetLng, int k, Predicate<Restaurant> filter) {
        lastVisitedNodes = 0;
        if (root == null || k <= 0) {
            return Collections.emptyList();
        }
        if (!Double.isFinite(targetLat) || !Double.isFinite(targetLng)) {
            throw new IllegalArgumentException("Target coordinates must be finite numbers");
        }

        // Can never return more than n results, so cap k (also keeps the heap's initial capacity sensible).
        int effectiveK = Math.min(k, size);
        // Computed once per query (O(1)) instead of once per node: cos(lat) shrinks longitude degrees
        // to match latitude degrees in ground distance at the query's latitude.
        double lngScale = Math.cos(Math.toRadians(targetLat));

        // Max-heap ordered by distance: reversed() makes the LARGEST distance sit at the top.
        PriorityQueue<RestaurantDistance> maxHeap = new PriorityQueue<>(
                effectiveK,
                Comparator.comparingDouble(RestaurantDistance::distanceSq).reversed());

        searchRecursive(root, targetLat, targetLng, lngScale, effectiveK, maxHeap, filter);

        // A heap is only partially ordered, so copy it out and sort nearest-first: O(k log k).
        // Ties are broken by real distance, then by name, so the output order is deterministic.
        List<RestaurantDistance> results = new ArrayList<>(maxHeap);
        results.sort(Comparator.comparingDouble(RestaurantDistance::distanceSq)
                .thenComparingDouble(RestaurantDistance::haversineKm)
                .thenComparing(rd -> rd.restaurant().name()));
        // Read-only view so callers can't modify the result list.
        return Collections.unmodifiableList(results);
    }

    /**
     * Recursive branch-and-bound step for one node.
     *
     * Work per visited node: O(1) distance maths + O(1) filter test (cost of the predicate itself)
     * + O(log k) for a heap offer/poll. The number of visited nodes is what pruning reduces.
     */
    private void searchRecursive(KDNode node,
                                  double targetLat,
                                  double targetLng,
                                  double lngScale,
                                  int k,
                                  PriorityQueue<RestaurantDistance> maxHeap,
                                  Predicate<Restaurant> filter) {
        // Base case: fell off the tree.
        if (node == null) {
            return;
        }
        lastVisitedNodes++;

        // 1. Consider the point stored at this node as a candidate.
        Restaurant current = node.getData();
        double dLat = targetLat - current.lat();
        double dLng = (targetLng - current.lng()) * lngScale;
        double distSq = dLat * dLat + dLng * dLng;

        if (filter == null || filter.test(current)) {
            if (maxHeap.size() < k) {
                // Heap not full yet: every matching point is a candidate. O(log k).
                maxHeap.offer(toResult(current, distSq, targetLat, targetLng));
            } else if (distSq < maxHeap.peek().distanceSq()) {
                // Heap full: only keep this point if it beats the current worst (the top). Evict the
                // worst and insert the new one: O(log k) + O(log k).
                maxHeap.poll();
                maxHeap.offer(toResult(current, distSq, targetLat, targetLng));
            }
            // else: farther than all k current candidates → ignore.
        }

        // 2. Decide which child is on the target's side of this node's splitting plane.
        boolean splitsOnLat = node.axis() == AXIS_LAT;
        double targetCoord = splitsOnLat ? targetLat : targetLng;
        double nodeCoord = node.splitValue();
        double axisDiff = targetCoord - nodeCoord; // signed distance from the target to the plane

        KDNode nearSide = axisDiff < 0 ? node.getLeft() : node.getRight();
        KDNode farSide = axisDiff < 0 ? node.getRight() : node.getLeft();

        // 3. Always search the near side first — the closest points are most likely there.
        searchRecursive(nearSide, targetLat, targetLng, lngScale, k, maxHeap, filter);

        // 4. Bound check for the far side. Any point on the far side is at least |axisDiff| away along
        //    this axis, so planeDistSq is a lower bound on its distance (the other axis can only add to it).
        //    Longitude differences are scaled the same way as in the distance formula above so the bound
        //    is in the same units as the heap's distances.
        double scaledAxisDiff = splitsOnLat ? axisDiff : axisDiff * lngScale;
        double planeDistSq = scaledAxisDiff * scaledAxisDiff;
        // Pruning is only possible once k candidates exist; before that, every matching point still counts.
        boolean heapFull = maxHeap.size() == k;
        if (heapFull && planeDistSq >= maxHeap.peek().distanceSq()) {
            return; // Prune: the splitting plane is already farther than the current k-th best.
        }
        // 5. The far side might still hold something closer than the current k-th best: search it.
        searchRecursive(farSide, targetLat, targetLng, lngScale, k, maxHeap, filter);
    }

    /**
     * Wraps a candidate in a result object. The exact Haversine distance is only computed here, i.e.
     * for points that actually enter the heap, rather than for every visited node — trig is far more
     * expensive than the squared-distance arithmetic used for ranking.
     *
     * Time: O(1): one Haversine evaluation (a fixed number of operations) and one object creation.
     */
    private static RestaurantDistance toResult(Restaurant r, double distSq, double targetLat, double targetLng) {
        double km = Restaurant.calculateHaversineDistance(targetLat, targetLng, r.lat(), r.lng());
        return new RestaurantDistance(r, distSq, km);
    }

    /** @return number of restaurants indexed. Time: O(1) (cached during build). */
    public int size() {
        return size;
    }

    /** @return true when nothing is indexed. Time: O(1): compares the cached size with 0. */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Height is not cached, so it is recomputed by visiting every node. It is only called once per
     * city for the "Indexed ... height H" message, so the O(n) cost doesn't matter.
     *
     * Time: O(n). Space: O(h) recursion stack.
     *
     * @return number of levels in the tree (0 for an empty tree).
     */
    public int height() {
        return heightOf(root);
    }

    /** Height of a sub-tree = 1 + height of its taller child. Time: O(size of sub-tree): each node of the sub-tree is visited once. */
    private static int heightOf(KDNode node) {
        if (node == null) {
            return 0;
        }
        return 1 + Math.max(heightOf(node.getLeft()), heightOf(node.getRight()));
    }

    /**
     * Time: O(1): returns a counter.
     *
     * @return nodes examined by the most recent findKNearest call (pruning diagnostic).
     */
    public int getLastVisitedNodes() {
        return lastVisitedNodes;
    }
}
