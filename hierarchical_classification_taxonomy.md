# Hierarchical Classification Taxonomy for Memory Card Contexts

This document defines the hierarchical structure for classifying user messages into memory card contexts, entities, and sub-entities for information extraction and storage.

## Taxonomy Structure

**Format**: `context > entity > sub_entity`

**Example**: "My goal is to earn 120k$ next year" → `professional > professional_aspirations > compensation_expectations`

**Note**: The aspirational context has been absorbed into professional_aspirations (for career goals) and personal_goals (for life goals).

---

## 1. Professional Context

### 1.1 current_position
- **role** - Current job title/role
- **company** - Current employer
- **compensation** - Current salary and total comp
- **start_date** - When started current role

**Example messages:**
- "I'm currently a Senior Product Manager at Google"
- "I make $150k base salary"
- "I've been in this role for 2 years"

### 1.2 professional_experience
- **past_roles** - Previous job titles
- **past_companies** - Previous employers
- **responsibilities** - What they did in past roles
- **achievements** - Accomplishments and impact
- **duration** - How long in each role

**Example messages:**
- "I worked at Microsoft for 3 years as a PM"
- "At my last job, I launched a feature that increased revenue by 20%"
- "I led a team of 5 engineers"

### 1.3 awards
- **awards** - Professional awards and recognitions from employers or industry

**Example messages:**
- "I won Product Manager of the Year"
- "I received Employee of the Month at my company"
- "I was recognized as Top Performer in my division"

### 1.4 licenses_and_permits
- **driving_licenses** - Driver's licenses
- **professional_licenses** - Professional licenses (medical, law, CPA)
- **security_clearances** - Government clearances
- **operating_permits** - Other permits

**Example messages:**
- "I have a California driver's license"
- "I have a Secret security clearance"
- "My medical license expires next year"

### 1.5 professional_aspirations

#### 1.5.1 dream_roles 🎯 **(Consolidated: includes target companies & industries)**
- **desired_roles** - Roles they want to achieve
- **target_companies** - Specific companies they want to work at (merged from 1.5.2)
- **target_industries** - Industries they want to work in (merged from 1.5.3)
- **priority** - How important this role is
- **timeframe** - When they want to achieve it
- **readiness** - How ready they feel
- **skill_gaps** - Skills they need to develop
- **connections** - People they know at target companies
- **networking_actions** - Steps they're taking

**Example messages:**
- "I want to become a VP of Product in 2 years"
- "My dream is to be a CPO someday"
- "I'm aiming for a Director role next year"
- "I'd love to work at Stripe" ← target company
- "I have a friend at Airbnb who could refer me" ← connections
- "I'm attending Stripe Sessions to network" ← networking actions
- "I'm really interested in fintech" ← target industry
- "I want to transition into AI/ML" ← target industry
- "I want to be a Senior PM at Google in the fintech industry" ← role + company + industry

**Rationale for consolidation:**
In natural conversation, people often express dream roles, target companies, and target industries together (e.g., "I want to be a Senior PM at Google in fintech"). Separating these into distinct sub-entities created artificial boundaries. This consolidation better reflects how people naturally talk about career aspirations.

#### 1.5.2 compensation_expectations
- **target_salary** - Desired salary
- **minimum_acceptable** - Lowest they'd accept
- **stretch_goal** - Aspirational target
- **total_comp** - Including equity/bonus
- **flexibility** - How flexible they are
- **priorities** - What else matters (equity, benefits)

**Example messages:**
- "My goal is to earn 120k$ next year" ← KEY EXAMPLE
- "I need at least $180k base"
- "I want $350k total comp with equity"
- "I'm flexible on salary if the equity is good"

#### 1.5.3 desired_work_environment
- **work_mode** - Remote/hybrid/in-office
- **company_size** - Startup/scale-up/enterprise
- **company_stage** - Seed/Series A/public
- **management** - IC vs manager
- **culture_priorities** - Cultural attributes they want
- **deal_breakers** - Non-negotiables

**Example messages:**
- "I want to work hybrid, not fully remote"
- "I'm looking for a Series C startup"
- "I need work-life balance"
- "I won't work somewhere with on-call"

#### 1.5.4 career_change_considerations
- **considering_change** - Thinking about career change
- **change_type** - Role/industry/both
- **risk_tolerance** - Willingness to take risks
- **pay_cut** - Would they accept lower pay
- **obstacles** - What's blocking them
- **support_needed** - What help they need

**Example messages:**
- "I'm thinking about switching from engineering to PM"
- "I want to change industries but worried about the pay cut"
- "I need more executive presence for the VP role"

#### 1.5.5 job_search_status
- **currently_searching** - Actively looking or not
- **urgency** - How urgent the search is
- **applications** - How many applied
- **interviews** - Interviews in progress
- **offers** - Offers received
- **start_date** - When they want to start

**Example messages:**
- "I'm casually looking for new opportunities"
- "I've applied to 5 companies"
- "I have 2 interviews next week"
- "I got an offer from Google"

### 1.6 volunteer_experience
- **volunteer_roles** - Volunteer positions

**Example messages:**
- "I mentor junior PMs at Product School"

---

## 2. Learning Context

### 2.1 current_skills
- **skills** - Skills they currently have
- **proficiency** - How proficient they are
- **experience** - Years of experience
- **verification** - How the skill is verified

**Example messages:**
- "I'm an expert in product strategy"
- "I know Python at an intermediate level"
- "I have 8 years of data analysis experience"

### 2.2 languages
- **language** - Languages spoken
- **proficiency** - Proficiency level
- **certifications** - Language test scores/certs

**Example messages:**
- "I speak Spanish at B1 level"
- "I'm fluent in Mandarin"
- "I got a 110 on the TOEFL"

### 2.3 education_history
- **degrees** - Degrees earned
- **institutions** - Schools attended
- **field_of_study** - Major/specialization
- **gpa** - Academic performance
- **graduation_date** - When they graduated

**Example messages:**
- "I have a BS in Computer Science from Berkeley"
- "I graduated with a 3.7 GPA"
- "I studied Software Engineering"

### 2.4 learning_gaps ⭐ **(Grouped: skills & knowledge gaps)**

Missing skills and knowledge blocking career goals.

#### 2.4.1 skill_gaps
- **missing_skills** - Skills blocking career goals
- **impact** - How this gap affects them
- **blocking_aspiration** - The specific dream role, target skill, or career goal this gap prevents
- **aspiration_type** - Whether it blocks a role, skill, certification, or general career goal

**Example messages:**
- "I need to improve my executive presence to become a VP"
- "I lack experience managing people for the Director role"
- "I don't have enough technical skills for that Senior Engineer position"
- "I need better public speaking skills to get promoted"
- "I'm missing Python experience to transition into data science"

#### 2.4.2 knowledge_gaps
- **missing_knowledge** - Knowledge they need to develop
- **blocking_aspiration** - The specific dream role, target skill, or career goal this knowledge gap prevents
- **aspiration_type** - Whether it blocks a role, skill, certification, or general career goal

**Example messages:**
- "I don't understand blockchain well enough to work at Coinbase"
- "I need to learn more about AI/ML to transition into machine learning"
- "I lack fintech knowledge for the payments role I want"
- "I need deeper understanding of cloud architecture to become a Solutions Architect"

**Combined example:**
- "I need better executive presence and deeper fintech knowledge to become a VP at a payments company"

**Rationale for consolidation:**
Both skill gaps and knowledge gaps represent missing competencies that block career aspirations. They're often mentioned together and serve the same purpose: identifying what needs to be learned to achieve goals. Grouping them simplifies classification while maintaining distinct sub-entities for extraction.

### 2.5 learning_aspirations ⭐ **(Grouped: skills, education & certifications)**

Future learning goals across all domains.

#### 2.5.1 skill_aspirations
- **target_skills** - Skills they want to learn
- **learning_plan** - How they plan to learn
- **timeline** - When they want to learn it
- **progress** - Current progress

**Example messages:**
- "I want to learn machine learning"
- "I'm learning public speaking through Toastmasters"
- "I'm 25% done with my AI course"

#### 2.5.2 education_aspirations
- **desired_degrees** - Degrees they want to pursue
- **institutions** - Target schools
- **timeline** - When they plan to pursue
- **funding** - How they'll pay for it

**Example messages:**
- "I want to get an MBA from Stanford"
- "I'm planning to pursue a master's in AI"
- "I'll apply to business school in 2027"

#### 2.5.3 certification_aspirations
- **target_certs** - Certifications they want
- **study_plan** - How they're preparing
- **exam_date** - When they plan to take exam

**Example messages:**
- "I'm studying for the Google Cloud certification"
- "I want to get my PMP next year"
- "I'm taking a prep course for AWS"

**Combined example:**
- "I want to learn Python and get AWS certified while pursuing an MBA"

**Rationale for consolidation:**
All three types of learning aspirations (skills, education, certifications) are future-focused learning goals that naturally overlap and are often expressed together. Grouping them simplifies classification while maintaining distinct sub-entities for information extraction.

### 2.6 certifications
- **earned_certs** - Certifications they have
- **issue_date** - When they got it
- **expiry_date** - When it expires
- **status** - Active/expired

**Example messages:**
- "I'm AWS Solutions Architect certified"
- "I have my CSPO certification"
- "My PMP expires next year"

### 2.7 knowledge_areas
- **expertise_domains** - Broader knowledge areas

**Example messages:**
- "I'm knowledgeable about fintech and payments"
- "I have deep expertise in platform architecture"

### 2.8 learning_preferences
- **preferred_formats** - How they like to learn
- **pace** - Fast/slow learner
- **budget** - Learning budget
- **time_available** - Hours per week for learning

**Example messages:**
- "I learn best through hands-on projects"
- "I prefer books over videos"
- "I can dedicate 10 hours per week to learning"
- "I have $2000/year for courses"

### 2.9 learning_history
- **past_courses** - Courses they've taken
- **books** - Books they've read
- **outcomes** - What they learned

**Example messages:**
- "I took Andrew Ng's ML course on Coursera"
- "I read 'High Output Management'"
- "I completed a bootcamp on data science"

### 2.10 publications
- **publications** - Articles, papers, blog posts written

**Example messages:**
- "I wrote an article about platform products"
- "I published a paper on machine learning"
- "I maintain a technical blog about AI"

### 2.11 academic_awards
- **academic_awards** - Academic honors and recognitions from educational institutions

**Example messages:**
- "I made the Dean's List in college"
- "I graduated Summa Cum Laude"
- "I received the Outstanding Student Award"
- "I was inducted into Phi Beta Kappa"

---

## 3. Social Context

### 3.1 mentors
- **mentor_name** - Name of mentor
- **mentor_role** - Their role/title
- **relationship** - Formal/informal
- **frequency** - How often they meet
- **guidance_areas** - What they help with
- **impact** - How helpful they are

**Example messages:**
- "I have a mentor who's a VP at Stripe"
- "My mentor meets with me monthly"
- "Sarah helps me with leadership skills"

### 3.2 mentees
- **mentee_name** - Name of mentee
- **mentee_background** - Their background
- **guidance_provided** - What help is provided
- **progress** - How they're progressing

**Example messages:**
- "I'm mentoring a junior PM transitioning from design"
- "I help new PMs with product strategy"

### 3.3 professional_network
- **connections** - People in their network (peers, colleagues, acquaintances)
- **relationship_strength** - Strong/weak ties
- **interaction_frequency** - How often they interact
- **collaboration_type** - How they work together
- **last_interaction** - When they last connected
- **communities** - Professional communities and groups
- **community_type** - Online/in-person
- **membership_status** - Active/inactive member
- **engagement_level** - How engaged they are
- **community_value** - What they get from communities

**Example messages:**
- "I have an accountability partner for career goals"
- "I collaborate with other PMs in my Reforge cohort"
- "I know the Director of Product at Airbnb"
- "I have 450 LinkedIn connections"
- "I haven't talked to John in 6 months"
- "I'm in the Reforge community"
- "I attend SF Product Managers meetups"
- "I'm active in a Slack group for PMs"

### 3.4 recommendations
- **testimonial_from** - Who wrote the testimonial
- **testimonial_text** - The written recommendation
- **permission_to_share** - Can testimonial be shared publicly
- **reference_name** - Name of reference
- **reference_role** - Their title/position
- **relationship** - How they know each other

**Example messages:**
- "My manager wrote a strong LinkedIn recommendation"
- "My manager wrote me a strong recommendation"
- "I got a testimonial from a colleague"
- "John Smith can be a reference for me"
- "I can provide 3 professional references"
- "My former manager Sarah Johnson wrote a great LinkedIn recommendation and can serve as a reference"

**Rationale for consolidation:**
Testimonials (written endorsements) and references (people who can vouch for you) are two sides of the same coin - both are endorsements from others that validate your skills and experience. They're often mentioned together and serve the same purpose in career advancement.

### 3.5 networking ⭐ **(Grouped: activities, goals & preferences)**

Professional networking activities, goals, and preferences.

#### 3.5.1 networking_activities
- **activity_type** - Conference/coffee chat/etc.
- **date** - When it happened
- **people_met** - Who they met
- **follow_up** - Next steps

**Example messages:**
- "I attended ProductCon last month"
- "I had coffee with a PM at Google"
- "I'm going to Stripe Sessions in March"

#### 3.5.2 networking_goals
- **target_connections** - People they want to meet
- **target_events** - Events they want to attend
- **networking_strategy** - How they'll network

**Example messages:**
- "I want to meet more CPOs"
- "I'm planning to attend 3 conferences this year"

#### 3.5.3 networking_preferences
- **preferred_formats** - 1-on-1/groups/conferences
- **energy_impact** - Energizing/draining
- **style** - Approach to networking

**Example messages:**
- "I prefer 1-on-1 coffee chats over big events"
- "Networking drains me, I need recovery time"

**Combined example:**
- "I want to meet CTOs at AI conferences, preferably in small group settings"

**Rationale for consolidation:**
Networking activities (what they do), goals (who they want to meet), and preferences (how they like to network) are naturally interconnected aspects of professional networking. They're often mentioned together in conversation and together paint a complete picture of someone's networking approach and strategy.

---

## 4. Psychological Context

### 4.1 personality_profile
- **personality_type** - MBTI, Big Five, etc.
- **traits** - Key personality traits
- **self_description** - How they describe themselves

**Example messages:**
- "I'm an INTJ"
- "I'm introverted and analytical"
- "I'm a perfectionist"

### 4.2 values
- **professional_values** - What matters at work
- **priorities** - Value priorities

**Example messages:**
- "I value autonomy and impact"
- "Work-life balance is my top priority"
- "I care deeply about mission alignment"

### 4.3 motivations
- **intrinsic_motivations** - Internal motivators
- **extrinsic_motivations** - External motivators
- **demotivators** - What demotivates them

**Example messages:**
- "I'm motivated by solving hard problems"
- "Money isn't my main motivator"
- "Micromanagement kills my motivation"

### 4.4 working_style_preferences
- **work_style** - How they prefer to work
- **collaboration_style** - How they work with others
- **decision_making** - How they make decisions
- **communication_style** - How they communicate

**Example messages:**
- "I work best independently with clear goals"
- "I like collaborative brainstorming sessions"
- "I make decisions quickly based on data"

### 4.5 confidence_and_self_perception

#### 4.5.1 confidence_levels
- **overall_confidence** - General confidence level
- **confidence_changes** - How it's changed recently
- **domain_confidence** - Confidence by domain (technical, social, leadership, public speaking, career decisions)
- **confidence_factors** - What affects confidence

**Example messages:**
- "I'm feeling pretty confident lately (7/10)"
- "My confidence has decreased after the project failed"
- "I'm very confident technically (8/10)"
- "I struggle with public speaking (4/10)"
- "I'm not confident in my leadership abilities yet"

#### 4.5.2 imposter_syndrome_and_doubt
- **imposter_level** - How strong imposter feelings are
- **imposter_frequency** - How often they feel it
- **imposter_triggers** - What triggers imposter feelings
- **self_doubt_frequency** - How often they doubt themselves
- **doubt_situations** - When doubt appears
- **comparison_patterns** - How often they compare to others
- **self_efficacy** - Belief in abilities to succeed and grow
- **resilience** - Ability to bounce back from setbacks

**Example messages:**
- "I have moderate imposter syndrome"
- "I feel like a fraud when presenting to senior leaders"
- "I worry people will find out I'm not as good as they think"
- "I often doubt if I'm smart enough"
- "I worry I'll fail when taking on new challenges"
- "I constantly compare myself to peers"
- "I believe I can learn anything with effort"
- "I bounce back quickly from setbacks"

#### 4.5.3 self_talk_and_validation
- **inner_critic_strength** - How harsh self-talk is
- **self_compassion** - How kind to themselves
- **negative_thought_patterns** - Common negative thoughts
- **external_validation_need** - Need for external approval
- **internal_validation_ability** - Ability to self-validate
- **reaction_to_criticism** - How they handle criticism
- **reaction_to_praise** - How they handle praise

**Example messages:**
- "My inner critic is very harsh"
- "I beat myself up over mistakes"
- "I'm learning to be more self-compassionate"
- "I rely too much on others' approval"
- "I dismiss compliments"
- "I get defensive when criticized"

#### 4.5.4 confidence_building_strategies
- **strategies_that_help** - What builds confidence
- **strategies_that_hurt** - What hurts confidence
- **currently_working_on** - Current efforts to build confidence
- **confidence_goals** - Confidence goals
- **coping_strategies** - How they cope with low confidence

**Example messages:**
- "Keeping a wins journal helps my confidence"
- "Working with an executive coach on confidence"
- "My goal is to reach 8/10 confidence"
- "Talking to my mentor helps with imposter feelings"

### 4.6 career_decision_making_style
- **decision_style** - Analytical/intuitive/etc.
- **decision_factors** - What influences decisions
- **decision_confidence** - Confidence in decisions

**Example messages:**
- "I make career decisions based on data"
- "I trust my gut when choosing jobs"
- "I struggle with big career decisions"

### 4.7 work_environment_preferences
- **ideal_environment** - What environment they thrive in
- **stressors** - What stresses them at work
- **energizers** - What energizes them

**Example messages:**
- "I thrive in fast-paced environments"
- "Open offices stress me out"
- "I love collaborative projects"

### 4.8 stress_and_coping
- **stress_level** - Current stress level
- **stress_triggers** - What causes stress
- **coping_strategies** - How they cope
- **effectiveness** - What works/doesn't work

**Example messages:**
- "I'm pretty stressed right now (7/10)"
- "Tight deadlines stress me out"
- "Exercise helps me manage stress"

### 4.9 emotional_intelligence
- **self_awareness** - Understanding own emotions
- **empathy** - Understanding others' emotions
- **emotional_regulation** - Managing emotions

**Example messages:**
- "I'm very self-aware about my emotions"
- "I'm working on being more empathetic"

### 4.10 growth_mindset
- **mindset_level** - Fixed vs growth mindset
- **beliefs_about_talent** - Innate vs developed
- **approach_to_challenges** - How they approach challenges

**Example messages:**
- "I believe abilities can be developed with effort"
- "I see failures as learning opportunities"
- "I love challenging myself"

---

## 5. Personal Context

### 5.1 personal_life
- **life_stage** - Life stage (early career, mid-career, settling down, etc.)
- **age_range** - Age bracket
- **relationship_status** - Single/married/partnered/divorced
- **partner** - Partner's situation and career
- **children** - Kids and ages
- **dependents** - Other dependents (parents, family members)
- **childcare** - Childcare arrangements
- **family_support** - Support system (in-laws, relatives, friends)
- **life_transitions** - Recent or upcoming transitions (marriage, divorce, kids, empty nest)
- **life_priorities** - Current life priorities (family time, career focus, work-life balance)

**Example messages:**
- "I'm in my early 30s with a young family"
- "I just got married"
- "I'm prioritizing family time right now"
- "I'm married with a 1.5-year-old"
- "My spouse is a teacher"
- "My in-laws help with childcare"
- "I'm planning to have another child in 2-3 years"

### 5.2 health_and_wellbeing

#### 5.2.1 physical_health
- **overall_health** - General health status
- **chronic_conditions** - Ongoing health issues
- **energy_levels** - Energy/fatigue
- **limitations** - Physical limitations

**Example messages:**
- "I have a chronic back condition"
- "I'm sleep-deprived from new parent life"
- "I have low energy most days"

#### 5.2.2 mental_health
- **conditions** - Mental health conditions
- **severity** - How severe they are
- **treatment** - Treatment status
- **impact_on_work** - How it affects work

**Example messages:**
- "I have anxiety and it's managed with therapy"
- "I struggle with depression"
- "I'm on medication for ADHD"
- "Burnout is affecting my work performance"

#### 5.2.3 addictions_or_recovery
- **addiction_type** - Type of addiction
- **status** - Active/recovery
- **clean_since** - How long clean/sober
- **recovery_program** - AA/NA/etc.
- **support_system** - Support network
- **triggers** - What to avoid
- **impact_on_career** - Career implications

**Example messages:**
- "I'm 9 months sober from alcohol"
- "I attend AA meetings 3 times a week"
- "I can't attend after-work happy hours"
- "My sobriety is my top priority"

#### 5.2.4 overall_wellbeing
- **stress_level** - Current stress
- **wellbeing_score** - Overall wellbeing

**Example messages:**
- "I'm feeling pretty good overall (7/10)"
- "My stress level is high right now"

### 5.3 living_situation
- **housing_type** - Own/rent/etc.
- **location** - Where they live
- **living_with** - Who they live with
- **relocation_openness** - Willing to move
- **constraints** - What prevents relocation
- **home_office** - Remote work setup

**Example messages:**
- "I own a house in Austin"
- "I can't relocate because my partner has tenure"
- "I have a great home office setup"

### 5.4 financial_situation
- **stability** - Financial stability
- **debt** - Debt situation
- **emergency_fund** - Savings cushion
- **dependents** - Financial dependents
- **income_dependency** - Single/dual income
- **risk_tolerance** - Financial risk tolerance
- **stress_level** - Financial stress

**Example messages:**
- "I have $45k in student loan debt"
- "I can't afford career risks right now"
- "I have 3-6 months emergency fund"
- "I'm financially stressed about the mortgage"

### 5.5 personal_goals
- **non_career_goals** - Personal life goals
- **category** - Health/family/relationship/etc.
- **priority** - Importance level
- **timeframe** - When they want to achieve it
- **progress** - Current progress

**Example messages:**
- "I want to maintain sobriety (highest priority)"
- "I want to lose 20 pounds in 6 months"
- "I want to be more present with my family"
- "I want to take my partner on an anniversary trip"
- "I want to get a dog in 1-2 years"

### 5.6 personal_projects
- **project_name** - Name of personal/side projects (both career-relevant and hobbies)
- **project_description** - What the project does
- **project_type** - Career-related/hobby/creative/etc.
- **project_role** - Their role in the project
- **project_skills** - Skills used (if applicable)
- **time_commitment** - Hours per week
- **motivation** - Why they do it

**Example messages:**
- "I built an open-source analytics dashboard"
- "I maintain a product management blog with 10k readers"
- "I'm restoring a vintage motorcycle with my dad"
- "I have a vegetable garden in my backyard"
- "I spend 3 hours/week on my project"

### 5.7 lifestyle_preferences
- **work_life_balance** - How important it is
- **ideal_schedule** - Preferred work schedule
- **flexibility_needs** - What flexibility they need
- **non_negotiables** - What they won't compromise

**Example messages:**
- "Work-life balance is critical (10/10 importance)"
- "I need flexibility for AA meetings"
- "I won't work more than 45 hours/week"
- "Remote work is non-negotiable"

### 5.8 life_constraints
- **constraint_type** - Family/health/location/financial
- **description** - What the constraint is
- **impact_on_career** - How it affects career
- **severity** - How limiting it is
- **timeframe** - How long it will last

**Example messages:**
- "I can't travel because of childcare responsibilities"
- "I need to stay near my mother for medical support"
- "I can't afford to take a pay cut"
- "My recovery meetings limit evening availability"

### 5.9 life_enablers
- **enabler_type** - Family/support/location/etc.
- **description** - What helps them
- **benefit_to_career** - How it helps career
- **strength** - How strong the enabler is

**Example messages:**
- "My in-laws provide free childcare"
- "My spouse is very supportive of my career"
- "My AA community keeps me accountable"

### 5.10 major_life_events
- **event_type** - Marriage/birth/move/health/etc.
- **date** - When it happened
- **description** - What happened
- **impact** - How it affected them

**Example messages:**
- "I got married last year"
- "My first child was born in 2023"
- "I started recovery 9 months ago"
- "We bought our first house"

### 5.11 personal_values
- **life_values** - What matters in life
- **importance** - Priority level

**Example messages:**
- "Family is my top priority"
- "Health and sobriety are most important"
- "I value authenticity and honesty"

### 5.12 life_satisfaction
- **overall_satisfaction** - Overall life satisfaction
- **satisfaction_by_area** - Breakdown by area
- **areas_to_improve** - What they want to improve

**Example messages:**
- "I'm satisfied with life overall (7/10)"
- "I'm very happy with my family (9/10)"
- "I want to improve my work fulfillment"
- "I'm dissatisfied with my social connections (5/10)"

---

## Usage for Hierarchical Classification

### Training Data Format

For training the hierarchical BERT classifier, create training data in this format:

```csv
message,context,entity,sub_entity
"My goal is to earn 120k$ next year",professional,professional_aspirations,compensation_expectations
"I'm 9 months sober from alcohol",personal,health_and_wellbeing,addictions_or_recovery
"I want to become a VP of Product",professional,professional_aspirations,dream_roles
"I'm an expert in product strategy",learning,current_skills,skills
"I have moderate imposter syndrome",psychological,confidence_and_self_perception,imposter_syndrome
"I'm married with a young child",personal,family_situation,children
```

### Classification Flow

1. **Level 1: Context Classification** (5 classes)
   - professional, learning, social, psychological, personal

2. **Level 2: Entity Classification** (context-specific)
   - For professional: current_position, professional_experience, professional_aspirations, etc.
   - For personal: family_situation, health_and_wellbeing, life_stage, etc.

3. **Level 3: Sub-entity Classification** (entity-specific)
   - For professional_aspirations: dream_roles (includes companies & industries), compensation_expectations, desired_work_environment, etc.
   - For health_and_wellbeing: physical_health, mental_health, addictions_or_recovery

### Routing for Information Extraction

Once classified, route to the appropriate extraction prompt and API endpoint:

```
Message: "My goal is to earn 120k$ next year"
↓
Context: professional
↓
Entity: professional_aspirations
↓
Sub-entity: compensation_expectations
↓
Extract: {target_base_salary: 120000, currency: "USD", timeframe: "next year"}
↓
Store: POST /api/harmonia/professional/professional-aspirations
```

---

## Notes

- Some messages may map to multiple paths (e.g., "I want to be a VP and earn $300k" maps to both dream_roles and compensation_expectations)
- The hierarchical structure allows for graceful degradation: if sub-entity classification is uncertain, fall back to entity level
- Context boundaries may overlap (e.g., "I want work-life balance" could be psychological > values OR personal > lifestyle_preferences)
- Always prefer the most specific classification level possible for better information extraction
