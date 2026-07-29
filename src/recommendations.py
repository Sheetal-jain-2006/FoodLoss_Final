def get_recommendations(raw_input, predicted_class, predicted_prob):
    """
    Generates actionable, parameter-driven recommendations to reduce food loss risk
    based on shipment features and model predictions.
    
    Parameters:
    - raw_input: dict, contains raw physical values entered by the user
    - predicted_class: str, ('Low Risk', 'Medium Risk', 'High Risk')
    - predicted_prob: dict, class probabilities
    
    Returns:
    - dict, containing risk summary and a list of specific recommendation strings
    """
    recommendations = []
    
    # 1. Temperature Recommendations
    temp = raw_input.get("Storage_Temperature", 0.0)
    crop = raw_input.get("Crop_Type", "Wheat")
    
    # Crop-specific storage temperature recommendations
    if crop == "Wheat":
        ideal_temp = "10-15°C"
        temp_thresh = 20.0
    elif crop == "Corn":
        ideal_temp = "8-12°C"
        temp_thresh = 18.0
    else: # Rice
        ideal_temp = "12-15°C"
        temp_thresh = 18.0
        
    if temp > temp_thresh:
        recommendations.append(
            f"High storage temperature detected ({temp:.1f}°C). For {crop}, "
            f"maintain storage between {ideal_temp} using active cooling or aeration."
        )
    elif temp < 0:
        recommendations.append(
            f"Freezing storage temperature detected ({temp:.1f}°C). Protect crops "
            f"from frost damage by adjusting thermostat set points."
        )
        
    # 2. Humidity Recommendations
    humidity = raw_input.get("Storage_Humidity", 0.0)
    if humidity > 75.0:
        recommendations.append(
            f"High storage humidity detected ({humidity:.1f}%). This significantly increases "
            f"mold, fungal growth, and spoilage. Increase ventilation or use active dehumidifiers."
        )
    elif humidity < 40.0:
        recommendations.append(
            f"Low storage humidity detected ({humidity:.1f}%). This can lead to crop moisture "
            f"loss and weight reduction. Monitor crop quality."
        )
        
    # 3. Storage Duration Recommendations
    storage_time = raw_input.get("Warehouse_Storage_Time", 0.0)
    if storage_time > 15.0: # assuming days or normalized units
        recommendations.append(
            f"Excessive warehouse storage time ({storage_time:.1f} days). Prioritize "
            f"first-in-first-out (FIFO) inventory dispatching to avoid degradation."
        )
        
    # 4. Delivery and Transport Duration
    delivery_time = raw_input.get("Delivery_Time", 0.0)
    if delivery_time > 24.0: # hours
        recommendations.append(
            f"Extended transit duration ({delivery_time:.1f} hours). Use temperature-controlled "
            f"vehicles (reefers) to prevent heat build-up inside the vehicle."
        )
        
    # 5. Route Distance
    distance = raw_input.get("Route_Distance", 0.0)
    if distance > 500.0: # km
        recommendations.append(
            f"Long transportation route ({int(round(distance))} km). Consider using regional hubs, "
            f"splitting shipments, or consolidating logistics to reduce single-trip risk."
        )
        
    # 6. Vibration Levels
    vibration = raw_input.get("Vibration_Level", 0.0)
    if vibration > 250.0:
        recommendations.append(
            f"High vehicle vibration level detected ({vibration:.1f} m/s²). Inspect truck "
            f"suspension and secure packaging using shock-absorbing pallets to prevent bruising."
        )
        
    # 7. Weather Impact
    weather = raw_input.get("Weather_Impact", 0.0)
    if weather > 5.0: # Assuming scale of 0 to 10
        recommendations.append(
            f"Severe weather impact index predicted ({weather:.1f}/10). Use reinforced, "
            f"moisture-resistant tarp covers and plan transit during milder weather windows."
        )
        
    # 8. Unloading / Terminal Queue Time
    queue = raw_input.get("Queue_Time", 0.0)
    if queue > 4.0: # hours
        recommendations.append(
            f"Long terminal/unloading queue time ({queue:.1f} hours) at receiving station. "
            f"Implement appointment scheduling or pre-clearance to avoid sun exposure on docks."
        )
        
    # 9. Quality Ratio
    quality_ratio = raw_input.get("Quality_Maintenance_Ratio", 100.0)
    if quality_ratio < 50.0:
        recommendations.append(
            f"Low Quality Maintenance Ratio ({quality_ratio:.1f}%). Sort and isolate "
            f"spoiled portions immediately to prevent cross-contamination of the batch."
        )
        
    # Overall summary recommendation based on model predictions
    if predicted_class == "High Risk":
        summary = "CRITICAL ACTION REQUIRED: Environment and logistics factors indicate extreme risk of crop loss. Take immediate corrective measures."
        recommendations.insert(0, summary)
    elif predicted_class == "Medium Risk":
        summary = "MONITORING WARNING: Moderate spoilage risk. Audit temperature settings and expedite transit schedules."
        recommendations.insert(0, summary)
    else:
        if len(recommendations) == 0:
            recommendations.append("All logistics and environmental parameters are within safe ranges.")
        summary = "OPTIMAL CONDITIONS: Low spoilage risk. Maintain current shipment profile."
        recommendations.insert(0, summary)
        
    return {
        'risk_summary': summary,
        'recommendations': recommendations
    }
