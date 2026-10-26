package com.spatial.restaurantfinder.service;

import com.google.maps.GeoApiContext;
import com.google.maps.GeocodingApi;
import com.google.maps.PlacesApi;
import com.google.maps.errors.ApiException;
import com.google.maps.errors.InvalidRequestException;
import com.google.maps.errors.OverDailyLimitException;
import com.google.maps.errors.OverQueryLimitException;
import com.google.maps.errors.RequestDeniedException;
import com.google.maps.errors.ZeroResultsException;
import com.google.maps.model.GeocodingResult;
import com.google.maps.model.LatLng;
import com.google.maps.model.PlaceType;
import com.google.maps.model.PlacesSearchResponse;
import com.google.maps.model.PlacesSearchResult;
import com.spatial.restaurantfinder.model.Restaurant;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

/**
 * Thin HTTPS client for geocoding addresses and fetching nearby restaurants from Google.
 *
 * The API key is read from the GOOGLE_MAPS_API_KEY environment variable, then from a .env file in the
 * working directory; if neither exists, construction fails with an IllegalStateException that explains
 * how to set it. The class is AutoCloseable so the HTTP threads are released by try-with-resources.
 *
 * TIME COMPLEXITY ANALYSIS (n = places collected, at most 40; F = lines in a .env file)
 *
 * Summary
 *   One search: Θ(n) expected local processing plus a constant number of HTTP requests
 *   (2 on the Places API (New) path, at most 3 on the Legacy fallback path).
 *
 * Reasoning in plain terms:
 *   1. The number of requests does not depend on n. The New path sends two fixed queries (vegetarian and
 *      non-vegetarian); the Legacy path follows at most three result pages, with a bounded number of retries.
 *   2. Each response is parsed in a single pass, visiting every place a constant number of times, so
 *      parsing is O(n) overall.
 *   3. Places from the different queries are merged by place id in a LinkedHashMap. A hash-map insert takes
 *      O(1) on average, so merging n places is O(n) expected. Comparing every pair of places instead would
 *      cost O(n²).
 *   4. Every returned place must be read at least once, so Ω(n) is unavoidable and the processing is optimal.
 *   5. The time a user actually waits is dominated by network latency (hundreds of milliseconds per request),
 *      not by this CPU work. Looking up the API key reads at most three small files, O(F).
 */
public final class GooglePlacesClient implements AutoCloseable {

    public static final String API_KEY_NAME = "GOOGLE_MAPS_API_KEY";
    /** Google's maximum search radius for Nearby Search is 50 km. */
    public static final int MAX_RADIUS_METERS = 50_000;
    public static final int DEFAULT_MAX_RESULTS = 60;
    /** Google returns at most 20 results per page and at most 3 pages for Nearby Search. */
    private static final int PAGE_SIZE = 20;
    private static final int MAX_PAGES = 3;

    /** A next_page_token takes a short moment to become valid on Google's side. */
    private static final long NEXT_PAGE_INITIAL_DELAY_MS = 2_000L;
    private static final long NEXT_PAGE_RETRY_DELAY_MS = 1_500L;
    private static final int NEXT_PAGE_MAX_ATTEMPTS = 5;

    private static final Logger LOG = LoggerFactory.getLogger(GooglePlacesClient.class);

    /** Shared, thread-safe client from the Google Maps Java library (used for geocoding + legacy Places). */
    private final GeoApiContext context;
    /** Raw key, needed for the Places API (New) calls which are made with Java's own HttpClient. */
    private final String rawApiKey;

    /**
     * Creates a client using the API key resolved from the environment or .env.
     * Time: O(lines in .env) for key lookup: the file is read once, line by line.
     */
    public GooglePlacesClient() {
        this(resolveApiKey());
    }

    /**
     * Creates a client using an explicit API key.
     *
     * Timeouts stop the CLI from hanging forever on a bad network; maxRetries retries
     * transient failures automatically; queryRateLimit(10) caps the client at 10 requests/second so it
     * does not exceed Google's per-second quota. Time: O(1): building the client makes no network call.
     */
    public GooglePlacesClient(String apiKey) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Google Maps API key is empty.");
        }
        this.rawApiKey = apiKey.strip();
        this.context = new GeoApiContext.Builder()
                .apiKey(this.rawApiKey)
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(15, TimeUnit.SECONDS)
                .writeTimeout(15, TimeUnit.SECONDS)
                .retryTimeout(30, TimeUnit.SECONDS)
                .maxRetries(3)
                .queryRateLimit(10)
                .build();
    }

    // ------------------------------------------------------------------------------------------
    // API key resolution
    // ------------------------------------------------------------------------------------------

    /**
     * Resolves the API key from the environment, then from ./.env.
     *
     * Placeholder values copied from .env.example ("your-api-key-here") are treated as
     * missing so the user gets the helpful "not found" message instead of a confusing
     * "request denied" from Google.
     *
     * Time: O(F) where F is the total number of lines across the (at most 3) candidate .env files.
     * Space: O(F) because each file is read fully into memory (they are tiny).
     *
     * @throws IllegalStateException when no key can be found
     */
    public static String resolveApiKey() {
        // 1. Environment variable wins, so the key can be set without creating a file.
        String fromEnv = System.getenv(API_KEY_NAME);
        if (fromEnv != null && !fromEnv.isBlank()
                && !fromEnv.equalsIgnoreCase("your-api-key-here")
                && !fromEnv.equalsIgnoreCase("your_api_key_here")) {
            return fromEnv.strip();
        }

        // 2. Look for a .env file in a few likely places relative to where the program was started,
        //    so it works regardless of whether the program is started from the project folder, its parent, or a sub-folder.
        Path cwd = Paths.get(System.getProperty("user.dir", "."));
        List<Path> candidateEnvFiles = List.of(
                cwd.resolve(".env"),
                cwd.resolve("restaurant-finder").resolve(".env"),
                cwd.resolve("..").resolve(".env")
        );

        for (Path envFile : candidateEnvFiles) {
            Optional<String> fromDotEnv = readKeyFromDotEnv(envFile);
            if (fromDotEnv.isPresent()) {
                return fromDotEnv.get();
            }
        }

        // 3. Nothing found: explain exactly how to fix it.
        throw new IllegalStateException(
                API_KEY_NAME + " not found.\n"
                        + "  Set it as an environment variable, e.g.\n"
                        + "    export " + API_KEY_NAME + "=\"your-key\"          (macOS / Linux)\n"
                        + "    $env:" + API_KEY_NAME + "=\"your-key\"           (Windows PowerShell)\n"
                        + "  or create a .env file containing:\n"
                        + "    " + API_KEY_NAME + "=your-key");
    }

    /**
     * Minimal .env parser: supports KEY=VALUE, an optional export  prefix,
     * single or double quotes around the value, blank lines and # comments.
     *
     * Written by hand rather than pulling in a dotenv library, since only one key is needed.
     *
     * Time: O(F) for F lines (each line is processed in time linear in its length).
     * Space: O(F) for the lines read from disk.
     */
    static Optional<String> readKeyFromDotEnv(Path envFile) {
        if (envFile == null || !Files.isRegularFile(envFile) || !Files.isReadable(envFile)) {
            return Optional.empty();
        }
        List<String> lines;
        try {
            lines = Files.readAllLines(envFile, StandardCharsets.UTF_8);
        } catch (IOException e) {
            // An unreadable file shouldn't crash the app; just try the next candidate.
            LOG.warn("Could not read {}: {}", envFile, e.getMessage());
            return Optional.empty();
        }

        for (String rawLine : lines) {
            String line = rawLine.strip();
            if (line.isEmpty() || line.startsWith("#")) {
                continue; // blank line or comment
            }
            if (line.startsWith("export ")) {
                line = line.substring("export ".length()).strip(); // shell-style "export KEY=VALUE"
            }
            int eq = line.indexOf('=');
            if (eq <= 0) {
                continue; // no '=' or no key name before it → not a KEY=VALUE line
            }
            String key = line.substring(0, eq).strip();
            if (!API_KEY_NAME.equals(key)) {
                continue; // some other variable
            }
            String value = line.substring(eq + 1).strip();
            value = stripQuotesAndComment(value);
            if (!value.isEmpty() && !value.equalsIgnoreCase("your-api-key-here") && !value.equalsIgnoreCase("your_api_key_here")) {
                return Optional.of(value);
            }
        }
        return Optional.empty();
    }

    /**
     * Removes surrounding quotes ("abc" or 'abc') or a trailing # comment.
     * Time: O(length of value): the value is scanned once for a closing quote or a comment marker.
     */
    private static String stripQuotesAndComment(String value) {
        if (value.length() >= 2) {
            char first = value.charAt(0);
            if (first == '"' || first == '\'') {
                int closing = value.indexOf(first, 1);
                if (closing > 0) {
                    // Quoted value: everything between the quotes, anything after is ignored.
                    return value.substring(1, closing).strip();
                }
            }
        }
        // Unquoted value: " #" starts an inline comment.
        int hash = value.indexOf(" #");
        if (hash >= 0) {
            value = value.substring(0, hash);
        }
        return value.strip();
    }

    // ------------------------------------------------------------------------------------------
    // Geocoding
    // ------------------------------------------------------------------------------------------

    /**
     * Resolves an address, city or landmark to coordinates using the Geocoding API.
     *
     * Google may return several candidates; the first one is Google's best match, so it is used.
     *
     * Time: 1 HTTP request (R = 1, plus library retries on failure) + O(1) local work.
     *
     * @throws PlacesClientException when the address cannot be resolved or the API call fails
     */
    public LatLng geocodeAddress(String address) {
        if (address == null || address.isBlank()) {
            throw new IllegalArgumentException("Address must not be blank");
        }
        try {
            // .await() blocks until the HTTP response arrives (synchronous call).
            GeocodingResult[] results = GeocodingApi.geocode(context, address.strip()).await();
            if (results == null || results.length == 0 || results[0].geometry == null) {
                throw new PlacesClientException("No location found for \"" + address.strip() + "\".");
            }
            GeocodingResult best = results[0];
            LOG.debug("Geocoded '{}' -> {} ({})", address, best.geometry.location, best.formattedAddress);
            return best.geometry.location;
        } catch (ZeroResultsException e) {
            throw new PlacesClientException("No location found for \"" + address.strip() + "\".", e);
        } catch (ApiException e) {
            throw translate("Geocoding", e);
        } catch (InterruptedException e) {
            // Restore the interrupt flag so code higher up can still see the thread was interrupted.
            Thread.currentThread().interrupt();
            throw new PlacesClientException("Geocoding was interrupted.", e);
        } catch (IOException e) {
            throw new PlacesClientException("Network error while geocoding: " + e.getMessage(), e);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Nearby search
    // ------------------------------------------------------------------------------------------

    /** Fetches restaurants around a coordinate using the default result limit (DEFAULT_MAX_RESULTS). */
    public List<Restaurant> fetchNearbyRestaurants(double lat, double lng, int radiusMeters) {
        return fetchNearbyRestaurants(lat, lng, radiusMeters, DEFAULT_MAX_RESULTS);
    }

    /**
     * Fetches restaurants around a coordinate. The Places API (New) is tried first and returns up to 40
     * places (two queries of at most 20 each; see fetchNearbyRestaurantsNew). If it fails, the Legacy
     * Places API is used instead; it follows next_page_token pagination until maxResults places are
     * collected or Google runs out of pages (max 3 pages × 20 results).
     *
     * Why two APIs? New Google Cloud projects can only use Places API (New); older projects may
     * only have the Legacy API enabled. Trying New first and falling back to Legacy means the app works
     * with either kind of key.
     *
     * Time: New path = 2 HTTP requests + O(P) parsing; Legacy fallback = up to 3 requests (with
     * ≥ 2 s waits between pages) + O(P) parsing.
     *
     * @param maxResults 1–60; only the Legacy fallback uses it (the New path always returns up to 40)
     * @throws PlacesClientException when both APIs fail; on the Legacy path, failures on later pages are
     *                               logged and the places collected so far are returned
     */
    public List<Restaurant> fetchNearbyRestaurants(double lat, double lng, int radiusMeters, int maxResults) {
        validateCoordinates(lat, lng);
        if (radiusMeters < 1 || radiusMeters > MAX_RADIUS_METERS) {
            throw new IllegalArgumentException(
                    "Radius must be between 1 and " + MAX_RADIUS_METERS + " metres, got " + radiusMeters);
        }

        // 1. Try modern Places API (New) first (required for all modern Google Cloud projects).
        try {
            List<Restaurant> results = fetchNearbyRestaurantsNew(lat, lng, radiusMeters, maxResults);
            LOG.debug("Places API (New) returned {} restaurant(s)", results.size());
            return results;
        } catch (PlacesClientException e) {
            // 2. Fall back to the Legacy API if the New one is not enabled / fails.
            LOG.warn("Places API (New) failed ({}); attempting Legacy Places API fallback...", e.getMessage());
            return fetchNearbyRestaurantsLegacy(lat, lng, radiusMeters, maxResults);
        }
    }

    /**
     * Calls Google Maps Places API (New) — POST https://places.googleapis.com/v1/places:searchNearby
     * Runs two searches: up to 20 vegetarian/vegan places, and up to 20 other restaurants.
     *
     * Why split by diet instead of one general query? Google ranks each query by popularity and returns
     * at most 20 places, so popular non-veg places would crowd small pure-veg places out of a single
     * top 20. Google has no "non-veg" place type, so the second query asks for restaurants and excludes
     * the vegetarian/vegan types. The two groups cannot overlap, and together they cover everything a plain
     * "restaurant" query would return, for the same number of requests. Places from the veg query are
     * marked veg, which is a reliable signal for the Pure Veg filter.
     *
     * The New API has no pagination, so this path returns at most 40 places and maxResults is not used.
     *
     * Results are collected in a LinkedHashMap keyed by place id. It keeps insertion order (non-veg
     * first) and stores an id that is returned twice only once, as a safeguard. Each insert is O(1) on
     * average, so merging is O(P) overall.
     *
     * Time: 2 HTTP requests + O(P) merge. Space: O(P).
     */
    private List<Restaurant> fetchNearbyRestaurantsNew(double lat, double lng, int radiusMeters, int maxResults) {
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(java.time.Duration.ofSeconds(10))
                .build();

        Map<String, Restaurant> merged = new LinkedHashMap<>();

        // 1. Non-veg side (up to 20): restaurants that Google does NOT tag as vegetarian/vegan.
        //    This is the primary query: if it fails, the exception makes the caller try the Legacy API.
        List<Restaurant> nonVeg = searchNearbyByType(client, lat, lng, radiusMeters,
                List.of("restaurant"), List.of("vegetarian_restaurant", "vegan_restaurant"), false, 20);
        for (Restaurant r : nonVeg) {
            merged.put(r.id(), r);
        }

        // 2. Veg side (up to 20): vegetarian or vegan restaurants.
        try {
            List<Restaurant> veg = searchNearbyByType(client, lat, lng, radiusMeters,
                    List.of("vegetarian_restaurant", "vegan_restaurant"), List.of(), true, 20);
            for (Restaurant r : veg) {
                // putIfAbsent: the two groups shouldn't overlap, but if one id repeats keep the first copy.
                merged.putIfAbsent(r.id(), r);
            }
        } catch (Exception e) {
            // The veg query is a "nice to have": if it fails the non-veg results are still returned.
            LOG.debug("Vegetarian query failed or empty: {}", e.getMessage());
        }

        return new ArrayList<>(merged.values());
    }

    /**
     * Performs one Places API (New) Nearby Search for the given included and excluded place types and
     * maps the JSON response into Restaurant objects. When vegQuery is true, every returned place is
     * marked as vegetarian.
     *
     * The X-Goog-FieldMask header asks Google for only the fields the client uses. This is required by
     * the New API and also lowers cost, because Google bills by the fields requested.
     *
     * Time: 1 HTTP request + O(P · t) to parse P places with t type tags each (t is small, so
     * effectively O(P)). Space: O(response size).
     */
    private List<Restaurant> searchNearbyByType(HttpClient client, double lat, double lng, int radiusMeters,
                                                List<String> includedTypes, List<String> excludedTypes,
                                                boolean vegQuery, int maxCount) {
        // Google allows 1..20 results per request; clamp to that range.
        int limit = Math.max(1, Math.min(maxCount, 20));
        // includedTypes matches places having ANY listed type; excludedTypes removes places having ANY
        // listed type. The type names are fixed constants, so joining them into JSON is safe.
        String included = "[\"" + String.join("\",\"", includedTypes) + "\"]";
        String excluded = excludedTypes.isEmpty()
                ? ""
                : "\"excludedTypes\":[\"" + String.join("\",\"", excludedTypes) + "\"],";
        // Locale.ROOT forces '.' as the decimal separator, otherwise some locales would write 12,97 and
        // produce invalid JSON.
        String jsonPayload = String.format(Locale.ROOT,
                "{\"includedTypes\":%s,%s\"maxResultCount\":%d,"
                + "\"locationRestriction\":{\"circle\":{\"center\":{\"latitude\":%.6f,\"longitude\":%.6f},\"radius\":%.1f}}}",
                included, excluded, limit, lat, lng, (double) radiusMeters);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://places.googleapis.com/v1/places:searchNearby"))
                .header("Content-Type", "application/json")
                .header("X-Goog-Api-Key", rawApiKey)
                .header("X-Goog-FieldMask", "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.types,places.primaryType,places.primaryTypeDisplayName,places.servesVegetarianFood")
                .timeout(java.time.Duration.ofSeconds(15))
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload, StandardCharsets.UTF_8))
                .build();

        try {
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() != 200) {
                // Non-200 (e.g. 403 when the New API isn't enabled) → caller falls back to Legacy.
                throw new PlacesClientException("Places API (New) HTTP " + response.statusCode() + ": " + response.body());
            }

            // Parsing the JSON body is O(response size).
            JsonObject root = JsonParser.parseString(response.body()).getAsJsonObject();
            if (!root.has("places") || root.get("places").isJsonNull()) {
                return new ArrayList<>(); // Google omits "places" entirely when nothing is found.
            }

            JsonArray places = root.getAsJsonArray("places");
            List<Restaurant> list = new ArrayList<>();
            for (JsonElement el : places) { // O(P)
                if (!el.isJsonObject()) continue;
                JsonObject p = el.getAsJsonObject();

                // Every field is optional in the response, so each one is read defensively with a fallback.
                String id = p.has("id") ? p.get("id").getAsString() : String.format(Locale.ROOT, "place@%.7f,%.7f", lat, lng);
                String name = "(unnamed)";
                if (p.has("displayName") && p.getAsJsonObject("displayName").has("text")) {
                    name = p.getAsJsonObject("displayName").get("text").getAsString();
                }
                String address = p.has("formattedAddress") ? p.get("formattedAddress").getAsString() : "";
                double rating = p.has("rating") ? p.get("rating").getAsDouble() : 0.0;

                double pLat = lat;
                double pLng = lng;
                if (p.has("location")) {
                    JsonObject loc = p.getAsJsonObject("location");
                    if (loc.has("latitude") && loc.has("longitude")) {
                        pLat = loc.get("latitude").getAsDouble();
                        pLng = loc.get("longitude").getAsDouble();
                    }
                }

                // A place returned by the vegetarian query is veg by definition; otherwise look for the tag.
                List<String> typesList = new ArrayList<>();
                boolean isVeg = vegQuery;
                if (p.has("types") && p.get("types").isJsonArray()) {
                    for (JsonElement te : p.getAsJsonArray("types")) { // O(t)
                        String t = te.getAsString();
                        typesList.add(t);
                        if ("vegetarian_restaurant".equalsIgnoreCase(t) || "vegan_restaurant".equalsIgnoreCase(t)) {
                            isVeg = true;
                        }
                    }
                }

                String primaryTypeDisplayName = null;
                if (p.has("primaryTypeDisplayName") && p.getAsJsonObject("primaryTypeDisplayName").has("text")) {
                    primaryTypeDisplayName = p.getAsJsonObject("primaryTypeDisplayName").get("text").getAsString();
                }
                String cuisine = Restaurant.resolveCuisine(primaryTypeDisplayName, typesList, name);

                try {
                    list.add(new Restaurant(id, name, pLat, pLng, rating, address, isVeg, typesList, cuisine));
                } catch (IllegalArgumentException e) {
                    // One bad record (e.g. invalid coordinates) shouldn't throw away the whole page.
                    LOG.debug("Skipping place {} with invalid data: {}", id, e.getMessage());
                }
            }
            return list;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new PlacesClientException("Nearby search was interrupted.", e);
        } catch (IOException e) {
            throw new PlacesClientException("Network error during nearby search: " + e.getMessage(), e);
        }
    }

    /**
     * Legacy Places API search (for older Google Cloud projects).
     *
     * Fetches page 1, then follows next_page_token for up to 3 pages in total, stopping early
     * once maxResults unique places are collected. Places are deduplicated by id with a
     * LinkedHashMap (O(1) average per insert, insertion order preserved).
     *
     * Time: up to 3 HTTP requests (with ≥ 2 s waits between pages, see fetchNextPage)
     * + O(P) processing. Space: O(P).
     */
    private List<Restaurant> fetchNearbyRestaurantsLegacy(double lat, double lng, int radiusMeters, int maxResults) {
        // Never ask for more than Google can return (3 pages × 20).
        int limit = Math.max(1, Math.min(maxResults, PAGE_SIZE * MAX_PAGES));

        Map<String, Restaurant> byId = new LinkedHashMap<>();
        LatLng location = new LatLng(lat, lng);

        // First page: errors here are fatal (there is nothing to return yet).
        PlacesSearchResponse page;
        try {
            page = PlacesApi.nearbySearchQuery(context, location)
                    .radius(radiusMeters)
                    .type(PlaceType.RESTAURANT)
                    .await();
        } catch (ZeroResultsException e) {
            return new ArrayList<>();
        } catch (ApiException e) {
            throw translate("Places Nearby Search", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new PlacesClientException("Nearby search was interrupted.", e);
        } catch (IOException e) {
            throw new PlacesClientException("Network error during nearby search: " + e.getMessage(), e);
        }

        int pagesFetched = 1;
        collect(page, byId, limit);

        // Follow pagination while more results are needed and Google reports further pages.
        while (byId.size() < limit
                && pagesFetched < MAX_PAGES
                && page != null
                && page.nextPageToken != null
                && !page.nextPageToken.isBlank()) {
            PlacesSearchResponse next = fetchNextPage(page.nextPageToken);
            if (next == null) {
                break; // later-page failures are non-fatal: keep what we have
            }
            page = next;
            pagesFetched++;
            collect(page, byId, limit);
        }

        LOG.debug("Fetched {} restaurant(s) across {} page(s)", byId.size(), pagesFetched);
        return new ArrayList<>(byId.values());
    }

    /**
     * Requests the next result page. Google rejects a token with INVALID_REQUEST until it becomes
     * active, so this waits and retries a few times before giving up.
     *
     * Time: at most NEXT_PAGE_MAX_ATTEMPTS HTTP requests; worst-case waiting is
     * 2 s + 4 × 1.5 s = 8 s of sleep. Space: O(P) for the page.
     *
     * @return the next page, or null when it cannot be fetched
     */
    private PlacesSearchResponse fetchNextPage(String token) {
        try {
            Thread.sleep(NEXT_PAGE_INITIAL_DELAY_MS); // token is never valid immediately
            for (int attempt = 1; attempt <= NEXT_PAGE_MAX_ATTEMPTS; attempt++) {
                try {
                    return PlacesApi.nearbySearchNextPage(context, token).await();
                } catch (InvalidRequestException e) {
                    // Token not active yet: wait a bit and try again (fixed-delay retry).
                    if (attempt == NEXT_PAGE_MAX_ATTEMPTS) {
                        LOG.warn("Next page token never became valid; keeping results collected so far.");
                        return null;
                    }
                    Thread.sleep(NEXT_PAGE_RETRY_DELAY_MS);
                }
            }
        } catch (ZeroResultsException e) {
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            LOG.warn("Pagination interrupted; keeping results collected so far.");
        } catch (ApiException | IOException e) {
            LOG.warn("Could not fetch next page ({}); keeping results collected so far.", e.getMessage());
        }
        return null;
    }

    /**
     * Adds the places from one page into sink, stopping as soon as limit is reached.
     * Time: O(P) for P results on the page (O(1) average per map insert).
     */
    private static void collect(PlacesSearchResponse page, Map<String, Restaurant> sink, int limit) {
        if (page == null || page.results == null) {
            return;
        }
        for (PlacesSearchResult result : page.results) {
            if (sink.size() >= limit) {
                return;
            }
            toRestaurant(result).ifPresent(r -> sink.putIfAbsent(r.id(), r));
        }
    }

    /**
     * Maps a Places result to the domain model, skipping entries without usable coordinates.
     * Permanently closed places are skipped too — no point recommending them.
     *
     * Time: O(t + L) (copying the types and resolving the cuisine).
     */
    static Optional<Restaurant> toRestaurant(PlacesSearchResult result) {
        if (result == null || result.geometry == null || result.geometry.location == null) {
            return Optional.empty();
        }
        if ("CLOSED_PERMANENTLY".equalsIgnoreCase(result.businessStatus)) {
            return Optional.empty();
        }
        double lat = result.geometry.location.lat;
        double lng = result.geometry.location.lng;
        // Use Google's place id when present; otherwise build a synthetic id from name + coordinates
        // so deduplication by id still works.
        String id = (result.placeId != null && !result.placeId.isBlank())
                ? result.placeId
                : String.format("%s@%.7f,%.7f", result.name, lat, lng);
        String address = result.vicinity != null && !result.vicinity.isBlank()
                ? result.vicinity
                : result.formattedAddress;
        List<String> typesList = (result.types != null) ? List.of(result.types) : Collections.emptyList();
        String cuisine = Restaurant.resolveCuisine(null, typesList, result.name);
        try {
            return Optional.of(new Restaurant(id, result.name, lat, lng, result.rating, address, false, typesList, cuisine));
        } catch (IllegalArgumentException e) {
            LOG.debug("Skipping place {} with invalid data: {}", id, e.getMessage());
            return Optional.empty();
        }
    }

    // ------------------------------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------------------------------

    /** Rejects impossible coordinates before spending an HTTP request on them. Time: O(1): four numeric comparisons. */
    private static void validateCoordinates(double lat, double lng) {
        if (!Double.isFinite(lat) || lat < -90.0 || lat > 90.0) {
            throw new IllegalArgumentException("Latitude must be between -90 and 90, got " + lat);
        }
        if (!Double.isFinite(lng) || lng < -180.0 || lng > 180.0) {
            throw new IllegalArgumentException("Longitude must be between -180 and 180, got " + lng);
        }
    }

    /**
     * Converts the library's checked ApiException subclasses into one unchecked exception with
     * a message a user can act on (e.g. "enable billing", "quota exceeded").
     * Time: O(length of the error message): the message is read once to look for the word "legacy".
     */
    private static PlacesClientException translate(String operation, ApiException e) {
        String detail = e.getMessage() == null ? "" : e.getMessage();
        if (e instanceof RequestDeniedException) {
            String hint = "Check that the key is valid, billing is enabled, and the Geocoding API and "
                    + "Places API are enabled for the key's Google Cloud project.";
            if (detail.toLowerCase().contains("legacy")) {
                hint = "This project cannot call Places API (Legacy), which this client library uses. "
                        + "Use a Google Cloud project that already has Places API (Legacy) enabled.";
            }
            return new PlacesClientException(operation + " request denied: " + detail + "\n  " + hint, e);
        }
        if (e instanceof OverQueryLimitException || e instanceof OverDailyLimitException) {
            return new PlacesClientException(operation + " quota exceeded: " + detail, e);
        }
        return new PlacesClientException(operation + " failed: " + detail, e);
    }

    /** Releases the HTTP client threads held by the underlying GeoApiContext. Time: O(1): no loops, it only shuts down the client's thread pool. */
    @Override
    public void close() {
        context.shutdown();
    }

    /**
     * Unchecked exception carrying a user-readable description of an API failure.
     * Unchecked (RuntimeException) so callers aren't forced to declare it on every method, while
     * Main can still catch it in one place and print the message.
     */
    public static final class PlacesClientException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        public PlacesClientException(String message) {
            super(message);
        }

        public PlacesClientException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
