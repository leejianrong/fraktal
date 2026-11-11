# City profiles for the three pilots (FRK-146, epic "City lookalikes at eye level")

Written 2026-11-12 from the private ground photos (`data/eval/ground-reference`, eval only, never
committed) by looking at all of them, plus class shares from the evaluation segmenter
([eval-segmentation.md](eval-segmentation.md)). The photos are Wikimedia Commons (2 to 3 per city) and
Mapillary (about 10 to 12 per city, none for Khartoum). Shares are mean over the photos, with n given;
under 8 photos treat them as hints.

## What the photos show

| | Khartoum + Bahri (n=3) | Manila + Quezon (n=14) | Nairobi centre (n=13) |
|---|---|---|---|
| Sky | pale dust haze, near white | blue with cloud, often partly blocked by wires | overcast and cool, thick cloud |
| Ground | bare tan earth, litter, concrete paving slabs, sealed road with low contrast | wide sealed grey asphalt, zebra crossings, painted lane lines, yellow kerb paint | grey asphalt with lane markings, paver-block and concrete pavements, black-white bollards |
| Buildings | 2 to 3 storeys, flat roofs, sand and cream, a minaret | 2 to 4 storeys, painted concrete (orange, teal, pink), corrugated sheet, hoarding fences in red and pink | 6 to 10 storeys, cream and grey concrete, glass, balconies, shopfront signage |
| Trees | almost none | heavy canopy, large broadleaf street trees | street trees, jacaranda, a few palms |
| Overhead | high light poles | dense cable tangles on poles | traffic lights on poles, wires on older streets, a flyover |
| Vehicles | white minibuses, pickups, coaches | motorbikes (often two-up), jeepneys, yellow delivery trucks | matatu minibuses, pickups, saloons |
| People | market crowds with colourful wraps | sparse, motorbike riders | dense pavement crowds in dark and bright clothes |
| Signs | very large billboards | election tarps, signboards on poles, street-name plates | many fascia and building signs, road signs |
| Drive side | right | right | left |

## Measured class shares (mean, sd), segmenter on photos

| Class | Khartoum (n=3) | Manila (n=14) | Nairobi centre (n=13) |
|---|---|---|---|
| building | 0.11 (0.04) | 0.09 (0.12) | 0.26 (0.18) |
| road | 0.26 (0.10) | 0.24 (0.13) | 0.24 (0.18) |
| pavement | 0.07 (0.07) | 0.02 (0.03) | 0.07 (0.12) |
| sky | 0.23 (0.17) | 0.16 (0.13) | 0.21 (0.18) |
| vegetation | 0.02 (0.02) | 0.34 (0.21) | 0.11 (0.12) |
| vehicle | 0.03 (0.02) | 0.08 (0.10) | 0.07 (0.10) |
| person | 0.20 (0.27) | 0.01 (0.01) | 0.02 (0.03) |

Khartoum's person share is one market photo. The photo spread is wide everywhere, so these are targets
to land inside, not numbers to hit.

## Gaps against the generator (what the first dense-urban strip showed)

The dense-urban strip against Manila-Quezon: vegetation 0.5% against 34%, building 49% against 5%,
road 5% against 24%, pavement 34% against 2% (our flat grey road reads as pavement), no vehicles in
frame against 8%, no fences against 4%. Common to all three cities and missing or weak today:

1. **Street trees and canopy** at eye level (Manila most, Nairobi next, Khartoum none).
2. **Road surface**: lane markings, zebra crossings, kerbs, and a clear difference between road and
   pavement. Our road is a flat grey.
3. **Street width and block height.** Dense-urban makes a tall narrow canyon; none of the three pilots
   looks like that. Manila is wide and low, Nairobi wide and mid-rise, Khartoum wide and low.
4. **Overhead utilities**: cable tangles (Manila), traffic lights and poles (Nairobi), high-mast lights
   (Khartoum). Card FRK-147.
5. **Vehicle and rider mix**: jeepneys and motorbikes (Manila), matatus (Nairobi), minibuses (Khartoum).
   Card FRK-147.
6. **Signage and hoardings**: scale and density differ a lot by city. Overlaps FRK-115.
7. **Sky and haze** per city (dust haze in Khartoum, cloud in Nairobi).
8. **Building palette**: Manila's painted pastels and orange, Nairobi's cream and grey, Khartoum's sand.

## What a profile has to carry (input to the profile layer)

Drive side; biome base and the parameters it overrides (street width, parcel size, storey profile,
footprint mix); building palette and materials; ground and pavement palette; tree species and density;
vehicle and rider mix; people density and clothing palette; signage and pole density; sky and haze;
the class-share targets above as a check, not an input.
