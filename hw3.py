class Features:
    def __init__(self, specs):
        # Specs is a dictionary mapping feature names to qualitative values
        # e.g., {'memory': 'HIGH', 'price': 'LOW', 'screen': 'COLOR'}
        self.specs = specs

class Device:
    def __init__(self, name, features):
        self.name = name
        self.features = features
        
    def __repr__(self):
        return f"Device({self.name})"

class Agent:
    def __init__(self, name):
        self.name = name
        self.goals = []           # e.g., 'gaming', 'budget', 'portability'
        self.stances = []         # Derived from goals: (Feature, PRO/CON, Value, Importance)
        self.history = {}         # e.g., {'HP': 'broke'}
        self.relationships = {}   # e.g., {'brother': 'Mac'}

    def add_goal(self, goal):
        self.goals.append(goal)
        self.infer_stances_from_goals()

    def infer_stances_from_goals(self):
        # Clear existing stances to rebuild them
        self.stances = []
        
        # Teleology: Inferring technical needs from user goals
        for goal in self.goals:
            if goal == 'gaming':
                self.stances.append(('memory', 'PRO', 'HIGH', 10))
                self.stances.append(('cpu', 'PRO', 'FAST', 9))
            elif goal == 'budget':
                self.stances.append(('price', 'PRO', 'LOW', 10))
                self.stances.append(('price', 'CON', 'HIGH', 10))
            elif goal == 'portability':
                self.stances.append(('weight', 'PRO', 'LIGHT', 8))
            elif goal == 'basic_web':
                self.stances.append(('price', 'PRO', 'LOW', 5))
                self.stances.append(('cpu', 'PRO', 'BASIC', 3))

def likes(agent, device):
    pro_reasons = []
    con_reasons = []
    
    # Check device specs against agent stances
    for feature, stance_type, expected_val, importance in agent.stances:
        actual_val = device.features.specs.get(feature)
        if actual_val:
            if stance_type == 'PRO' and actual_val == expected_val:
                pro_reasons.append((importance, f"It has {expected_val} {feature}."))
            elif stance_type == 'CON' and actual_val == expected_val:
                con_reasons.append((importance, f"It has {expected_val} {feature}, which you dislike."))
            elif stance_type == 'PRO' and actual_val != expected_val:
                con_reasons.append((importance, f"It lacks {expected_val} {feature}."))

    # Sort reasons by importance (highest first)
    pro_reasons.sort(reverse=True, key=lambda x: x[0])
    con_reasons.sort(reverse=True, key=lambda x: x[0])
    
    is_liked = len(pro_reasons) > len(con_reasons)
    
    explanation = f"Agent {agent.name} {'likes' if is_liked else 'dislikes'} the {device.name}.\n"
    if pro_reasons:
        explanation += "Pros:\n" + "\n".join([f"- {r[1]}" for r in pro_reasons]) + "\n"
    if con_reasons:
        explanation += "Cons:\n" + "\n".join([f"- {r[1]}" for r in con_reasons]) + "\n"
        
    return is_liked, explanation

def prefers(agent, devices):
    liked_devices = []
    explanations = []
    
    for device in devices:
        is_liked, expl = likes(agent, device)
        if is_liked:
            liked_devices.append(device)
            explanations.append(expl)
            
    if not liked_devices:
        return [], "No devices are preferred. None meet the requirements."
        
    return liked_devices, "\n---\n".join(explanations)

def recommend(agent, catalog):
    recommendations = []
    
    for device in catalog:
        score = 0
        reasons = []
        
        # 1. Norms (The salesperson knows best)
        screen_color = device.features.specs.get('screen')
        if screen_color == 'COLOR':
            score += 2
            reasons.append((2, "Norm: It has a color screen, which most people prefer."))
        elif screen_color == 'BW':
            score -= 5
            reasons.append((5, "Norm: Black and white screens are outdated."))

        # 2. History & Relationships
        if device.name in agent.history and agent.history[device.name] == 'broke':
            score -= 20
            reasons.append((20, f"History: You had a {device.name} before and it was unreliable."))
            
        if 'brother' in agent.relationships and agent.relationships['brother'] == device.name:
            score += 5
            reasons.append((5, f"Social: It is a {device.name} like your brother's, so he can help you."))

        # 3. Agent's explicit goals
        is_liked, expl = likes(agent, device)
        if is_liked:
            score += 10
            reasons.append((10, "It meets your primary technical goals."))
        else:
            score -= 10
            reasons.append((10, "It fails to meet your technical goals."))
            
        if score > 0:
            reasons.sort(reverse=True, key=lambda x: x[0])
            explanation = "\n".join([f"- {r[1]}" for r in reasons])
            recommendations.append((score, device, explanation))
            
    # Sort by highest score
    recommendations.sort(reverse=True, key=lambda x: x[0])
    
    if not recommendations:
        return "Recommendation: DO NOT BUY ANY COMPUTER. Explanation: None of the current market options satisfy your needs or overcome your negative history."
        
    top_score, top_device, top_expl = recommendations[0]
    return f"Highly Recommending: {top_device.name}\nReasons (in order of importance):\n{top_expl}"

# Create orthogonal fictional devices
mac_clone = Device("FruitBook Pro", Features({'memory': 'HIGH', 'cpu': 'FAST', 'price': 'HIGH', 'screen': 'COLOR', 'weight': 'LIGHT'}))
cheap_pc = Device("BudgetBox", Features({'memory': 'LOW', 'cpu': 'BASIC', 'price': 'LOW', 'screen': 'COLOR', 'weight': 'HEAVY'}))
retro_pc = Device("OldReliable", Features({'memory': 'LOW', 'cpu': 'BASIC', 'price': 'LOW', 'screen': 'BW', 'weight': 'HEAVY'}))

catalog = [mac_clone, cheap_pc, retro_pc]

# Taxonomy User 1: The Broke Gamer
gamer_student = Agent("Alex")
gamer_student.add_goal('gaming')
gamer_student.add_goal('budget')

# Taxonomy User 2: The Tech-Illiterate Parent
parent = Agent("Mom")
parent.add_goal('basic_web')
parent.relationships['brother'] = "FruitBook Pro" # Wants what her son has
parent.history['BudgetBox'] = 'broke'             # Hates the cheap PC brand

# Taxonomy User 3: Extreme Edge Case
luddite = Agent("Ron")
luddite.history['FruitBook Pro'] = 'broke'
luddite.history['BudgetBox'] = 'broke'
luddite.history['OldReliable'] = 'broke'

print("=== LIKES TEST ===")
is_liked, explanation = likes(gamer_student, mac_clone)
print(explanation)

print("\n=== PREFERS TEST ===")
pref_devices, pref_expl = prefers(gamer_student, catalog)
print(pref_expl)

print("\n=== RECOMMEND TEST: ALEX ===")
print(recommend(gamer_student, catalog))

print("\n=== RECOMMEND TEST: MOM ===")
print(recommend(parent, catalog))

print("\n=== RECOMMEND TEST: RON (NO COMPUTER) ===")
print(recommend(luddite, catalog))