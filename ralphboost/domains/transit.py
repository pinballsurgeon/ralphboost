import math
from .base import Domain

def _mean(values):
    return sum(values) / len(values) if values else 0.0

def _dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

class TransitDomain(Domain):
    def __init__(self, sample_rate=1.0, min_period=10.0, max_period=100.0, **kwargs):
        self.sample_rate = float(sample_rate)
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        # kwargs ignored for compatibility

    def get_context(self):
        return {
            "domain_name": "Kepler Transit Decomposition",
            "schema": """
Candidates must use one of these families:
1. "periodic_train": { "period": float, "phase": float, "width": float, "depth": float }
2. "local_dip": { "center": float, "width": float, "depth": float }
3. "baseline": { "intercept": float, "slope": float }
""",
            "hint": "Looking for exoplanet transits (periodic dips) and trends. Use 'periodic_train' for repeating dips."
        }

    def initialize_fitted(self, target):
        return [0.0] * len(target)

    def compute_residual(self, target, fitted):
        return [t - f for t, f in zip(target, fitted)]

    def energy(self, signal):
        return sum(x**2 for x in signal)

    def apply(self, component, current_fit, context=None):
        ctype = component.get("type")
        weight = float(component.get("weight", 1.0))
        n = len(current_fit)
        
        # All components are additive
        output = list(current_fit)
        
        if ctype == "baseline":
            intercept = float(component.get("intercept", 0.0))
            slope = float(component.get("slope", 0.0))
            for i in range(n):
                # Scale by weight? Typically weight scales the whole shape.
                # If intercept/slope are fitted to residual, then weight=1.0 reconstructs it.
                # If engine applies weight < 1.0 (shrinkage), we scale.
                # Base params are "full strength".
                val = (intercept + slope * i) * weight
                output[i] += val
                
        elif ctype == "periodic_train":
            period = float(component.get("period", 10.0))
            phase = float(component.get("phase", 0.0))
            width = float(component.get("width", 1.0))
            depth = float(component.get("depth", 0.0)) # Positive depth means dip
            
            # depth is magnitude. Dip is negative.
            # val = -depth * weight inside transit
            
            val = -depth * weight
            if period > 0:
                for i in range(n):
                    t = i  # Assuming unit steps for now, or use self.sample_rate logic if rigorous
                    # Transit happens when (t - phase) % period < width
                    # Centered? Let's say phase is start of transit.
                    # Or phase is center. Let's use phase as center.
                    # Normalized phase: (t - phase + P/2) % P - P/2
                    # distance from center
                    
                    # Simple logic: phase is start index? 
                    # Let's use: phase is the center of the first transit.
                    
                    dist = (t - phase + 0.5 * period) % period - 0.5 * period
                    if abs(dist) < 0.5 * width:
                        output[i] += val
                        
        elif ctype == "local_dip":
            center = float(component.get("center", 0.0))
            width = float(component.get("width", 1.0))
            depth = float(component.get("depth", 0.0))
            
            val = -depth * weight
            
            # Simple box dip
            start = center - 0.5 * width
            end = center + 0.5 * width
            
            start_idx = max(0, int(math.floor(start)))
            end_idx = min(n, int(math.ceil(end)))
            
            for i in range(start_idx, end_idx):
                # Sub-pixel? Let's just do binary inclusion for MVHR
                if start <= i <= end:
                    output[i] += val

        return output

    def propose(self, state, k=1, agent=None, context=None):
        residual = getattr(state, "residual", [])
        n = len(residual)
        if n == 0:
            return []
            
        candidates = []
        
        # 1. ALWAYS propose BaselineTrend
        # We don't fit it here, refine will fit it.
        candidates.append({
            "type": "baseline",
            "intercept": 0.0,
            "slope": 0.0
        })
        
        # 2. Propose Periodic Trains (Autocorrelation)
        # Simple coarse lag search
        # We want negative dips. Autocorr on raw residual might find positive correlation (matching shapes).
        # We need to find the PERIOD.
        
        # Compute autocorr for lags in [min_period, max_period]
        lags = []
        # optimization: only check integer lags or coarse grid
        min_lag = max(1, int(self.min_period))
        max_lag = min(n // 2, int(self.max_period))
        
        if max_lag > min_lag:
            # Calculate variance for normalization? 
            # Just dot product is enough for ranking
            
            # Coarse grid: step size 1
            for lag in range(min_lag, max_lag + 1):
                # sum(r[i] * r[i+lag])
                # Limit size for speed?
                s = 0.0
                count = 0
                for i in range(0, n - lag, 2): # stride 2 for speed
                    s += residual[i] * residual[i+lag]
                    count += 1
                if count > 0:
                    score = s / count
                    lags.append((score, lag))
            
            # Sort by score desc (positive correlation = repetition)
            lags.sort(key=lambda x: x[0], reverse=True)
            
            # Take top few distinct periods
            seen_periods = set()
            for score, lag in lags:
                if score <= 0: continue
                
                # Check if close to seen
                is_close = False
                for p in seen_periods:
                    if abs(p - lag) < 2.0: # simplistic tolerance
                        is_close = True
                        break
                if is_close:
                    continue
                    
                seen_periods.add(lag)
                candidates.append({
                    "type": "periodic_train",
                    "period": float(lag),
                    "phase": 0.0, # Will be found in refine
                    "width": 3.0, # Default width
                    "depth": 0.0  # Will be found in refine
                })
                if len(candidates) >= k + 1: # +1 for baseline
                    break

        # 3. Propose Local Dips (Rolling Min)
        # Find deepest points
        dips = []
        win = 5 # default scan window
        for i in range(n):
            if residual[i] < 0: # potential dip
                dips.append((residual[i], i))
        
        dips.sort(key=lambda x: x[0]) # Ascending (deepest negative first)
        
        seen_centers = set()
        for val, idx in dips:
            if len(candidates) >= k + 5: # Allow some overflow
                break
                
            # Check close
            is_close = False
            for c in seen_centers:
                if abs(c - idx) < 10.0:
                    is_close = True
                    break
            if is_close:
                continue
                
            seen_centers.add(idx)
            candidates.append({
                "type": "local_dip",
                "center": float(idx),
                "width": 3.0,
                "depth": 0.0 # Refine will fix
            })

        return candidates[:k]  # Respect k_candidates contract

    def refine(self, candidate, state):
        residual = getattr(state, "residual", [])
        n = len(residual)
        if n == 0:
            return candidate
            
        # DEBUG stats
        # r_min = min(residual)
        # r_max = max(residual)
        # r_mean = sum(residual)/n
        # print(f"DEBUG: refine residual stats: min={r_min:.2f} max={r_max:.2f} mean={r_mean:.2f}")

        ctype = candidate.get("type")
        
        if ctype == "baseline":
            # Linear regression: r = a + b*i
            # Minimize sum((r - (a + b*i))^2)
            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_xx = 0.0
            for i, r in enumerate(residual):
                sum_x += i
                sum_y += r
                sum_xy += i * r
                sum_xx += i * i
            
            denom = (n * sum_xx - sum_x * sum_x)
            if abs(denom) < 1e-9:
                return candidate # Singular
                
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            
            new_cand = dict(candidate)
            new_cand["intercept"] = intercept
            new_cand["slope"] = slope
            return new_cand
            
        elif ctype == "periodic_train":
            period = float(candidate.get("period", 10.0))
            width = float(candidate.get("width", 3.0))
            
            # We need to find Phase and Depth.
            # Scan phases [0, period)
            # For each phase, measure average depth inside transit vs outside?
            # Or just Analytic Depth Fit:
            # h_phase[i] = -1 if in transit else 0
            # alpha = (r . h) / (h . h)
            # alpha = sum(-r_in) / count_in
            # We maximize alpha (depth).
            
            best_phase = 0.0
            best_depth = 0.0
            best_sse = float('inf')
            
            # Coarse phase grid
            phase_steps = int(period)
            if phase_steps < 1: phase_steps = 1
            
            for p_idx in range(phase_steps):
                ph = float(p_idx)
                # Compute depth
                # Indices in transit:
                # (i - ph + P/2) % P - P/2 \in [-w/2, w/2]
                
                sum_r_in = 0.0
                count_in = 0
                
                # Optimized loop?
                # Just iterate all points? Slow for python?
                # For MVHR, simple is fine.
                
                # Construct template h (sparse logic?)
                # transit indices: k*period + ph - w/2 .. k*period + ph + w/2
                
                current_sum_r_in = 0.0
                current_count_in = 0
                
                # Iterate transits within range n
                # start_t such that start_t * P + ph - w/2 > 0
                
                num_transits = int(n / period) + 2
                for k in range(-1, num_transits):
                    center = k * period + ph
                    start = int(math.ceil(center - 0.5 * width))
                    end = int(math.floor(center + 0.5 * width))
                    
                    start = max(0, start)
                    end = min(n - 1, end)
                    
                    if start <= end:
                         # Slice sum?
                         # python list slice is fast enough
                         # sum(residual[start:end+1])
                         chunk = residual[start:end+1]
                         current_sum_r_in += sum(chunk)
                         current_count_in += len(chunk)
                
                if current_count_in > 0:
                    # alpha = sum(-r) / count
                    # -alpha = sum(r) / count (mean residual in transit)
                    # We want dip, so mean residual should be negative.
                    mean_r = current_sum_r_in / current_count_in
                    depth = -mean_r
                    
                    # if ph == 25.0:
                    #    print(f"DEBUG: Check ph=25.0: mean_r={mean_r:.2f} depth={depth:.2f} count={current_count_in}")

                    if depth > 0: # valid dip
                        # SSE approximation?
                        # Residual energy change ~ -alpha * sum(r . h) = -depth * sum(-r) = depth * sum(r) ?
                        # Actually we maximize correlation (depth * count * depth?)
                        # We want to maximize depth * sqrt(count) or similar?
                        # Just maximize depth for now (deepest dip)
                        if depth > best_depth:
                            best_depth = depth
                            best_phase = ph
            
            new_cand = dict(candidate)
            new_cand["phase"] = best_phase
            new_cand["depth"] = best_depth
            new_cand["width"] = width # Could optimize width too
            return new_cand
            
        elif ctype == "local_dip":
            center = float(candidate.get("center", 0.0))
            width = float(candidate.get("width", 3.0))
            
            # Optimize center locally +/- width
            search_range = int(width * 2)
            best_center = center
            best_depth = 0.0
            
            start_search = max(0, int(center - search_range))
            end_search = min(n, int(center + search_range))
            
            # Simple scan
            for c in range(start_search, end_search):
                start = int(c - 0.5 * width)
                end = int(c + 0.5 * width)
                start = max(0, start)
                end = min(n - 1, end)
                
                if start <= end:
                    chunk = residual[start:end+1]
                    s = sum(chunk)
                    count = len(chunk)
                    if count > 0:
                        depth = - (s / count)
                        if depth > best_depth:
                            best_depth = depth
                            best_center = float(c)
                            
            new_cand = dict(candidate)
            new_cand["center"] = best_center
            new_cand["depth"] = best_depth
            return new_cand

        return candidate

    def fingerprint(self, component, state=None):
        ctype = component.get("type")
        if ctype == "baseline":
            return "baseline"
        elif ctype == "periodic_train":
            p = float(component.get("period", 0))
            # Bucket period
            p_bucket = int(round(p * 10)) # 0.1 precision
            # Phase bucket?
            ph = float(component.get("phase", 0))
            ph_bucket = int(round(ph * 10))
            return ("periodic", p_bucket, ph_bucket)
        elif ctype == "local_dip":
            c = float(component.get("center", 0))
            c_bucket = int(round(c / 5.0)) # 5.0 bucket
            return ("local", c_bucket)
        return None
        
    def complexity(self, component, state=None):
        ctype = component.get("type")
        if ctype == "baseline":
            return 1.0
        elif ctype == "periodic_train":
            return 2.0 # Good explanation
        elif ctype == "local_dip":
            return 5.0 # Expensive explanation (don't spam these)
        return 1.0
