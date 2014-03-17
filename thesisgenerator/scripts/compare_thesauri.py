import os
import random
from discoutils.thesaurus_loader import Thesaurus
from thesisgenerator.utils.conf_file_utils import parse_config_file
import logging
'''
Samples and prints out the neighbours of randomly selected entries in a number of thesauri
'''
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
conf_files = ['conf/exp%d/exp%d_base.conf' % (x, x) for x in [6, 7, 8, 9, 10, 11, 58, 61]]
thes_files = []
for t in conf_files:
    conf, _ = parse_config_file(t)
    thes_files.append(conf['vector_sources']['unigram_paths'][0])
print thes_files
thesauri = [Thesaurus.from_tsv([t]) for t in thes_files]
entries = [set(t.keys()) for t in thesauri]
shared_entries = set.intersection(*entries)
print 'shared entries', len(shared_entries)
for e in random.sample(shared_entries, 20):
    print
    for thes, file in zip(thesauri, thes_files):
        print '{}, {}, {}'.format(os.path.basename(os.path.dirname(file)),
                        e,
                        ','.join([x[0] for x in thes[e][:5]]))




'''
thesauri exp10-12bAN_NN_gigaw-100_
4617 shared entries

Add, special/J_grant/N, special/J_dividend/N,special/J_bonus/N,special/J_distribution/N,special/J_feature/N,special/J_version/N
Mult, special/J_grant/N, special/J_technology/N,superior/J_character/N,special/J_safeguard/N,valiant/J_crew/N,larger/J_value/N
Left, special/J_grant/N, special/J_worry/N,special/J_payout/N,special/J_murphy/N,special/J_write-off/N,special/J_stock/N
Right, special/J_grant/N, lee/N_grant/N,peseta/N_grant/N,common/J_grant/N,faye/N_grant/N,hugh/J_grant/N
Baroni, special/J_grant/N, strong/J_power/N,little/J_flag/N,early/J_review/N,french/J_companion/N,total/J_france/N
Observed, special/J_grant/N, likely/J_bid/N,strong/J_power/N,impressive/J_shot/N,tiny/J_bit/N,flat/J_figure/N
APDT, special/J_grant/N, emotional/J_force/N,pharmaceutical/J_electronics/N,material/N_company/N,search/N_team/N,service/N_moratorium/N
Socher, special/J_grant/N, orphan/N_grant/N,special/J_account/N,common/J_grant/N,special/J_tariff/N,special/J_safeguard/N

Add, government/N_spending/N, manila/N_government/N,zimbabwe/N_government/N,bonn/N_government/N,government/N_practice/N,shanghai/N_government/N
Mult, government/N_spending/N, spending/N_company/N,company/N_profit/N,mod/N_squad/N,airline/N_profit/N,nec/N_production/N
Left, government/N_spending/N, government/N_organization/N,government/N_investigator/N,government/N_sale/N,government/N_cover-up/N,government/N_issue/N
Right, government/N_spending/N, stepped-up/J_spending/N,federal/J_spending/N,exploration/N_spending/N,domestic/J_spending/N,u.s./N_spending/N
Baroni, government/N_spending/N, public/J_spending/N,high/J_industry/N,domestic/J_market/N,financial/J_position/N,latest/J_attempt/N
Observed, government/N_spending/N, overall/J_market/N,public/J_spending/N,high/J_industry/N,domestic/J_market/N,textile/N_quota/N
APDT, government/N_spending/N, high/J_spending/N,card/N_bank/N,private/J_order/N,television/N_business/N,battery/N_business/N
Socher, government/N_spending/N, government/N_borrowing/N,public/J_spending/N,military/J_spending/N,federal/J_spending/N,advertising/N_spending/N

Add, public/J_relation/N, bilateral/J_relation/N,diplomatic/J_relation/N,stable/J_relation/N,formal/J_relation/N,strained/J_relation/N
Mult, public/J_relation/N, final/J_production/N,category/N_number/N,black/J_book/N,country/N_space/N,oakland/N_tribune/N
Left, public/J_relation/N, public/J_month/N,public/J_announcement/N,public/J_taste/N,public/J_devastation/N,public/J_domain/N
Right, public/J_relation/N, u.s.-canadian/J_relation/N,customer/N_relation/N,stable/J_relation/N,u.s.-japanese/J_relation/N,harmonious/J_relation/N
Baroni, public/J_relation/N, political/J_intrigue/N,new/J_share/N,investment/N_product/N,total/J_supply/N,domestic/J_drama/N
Observed, public/J_relation/N, political/J_intrigue/N,financiere/N_paribas/N,polish/J_economy/N,poorest/J_country/N,american/J_cinema/N
APDT, public/J_relation/N, capital/N_activity/N,exchange/N_operation/N,capital/N_expenditure/N,quarterly/J_distribution/N,property/N_operation/N
Socher, public/J_relation/N, foreign/J_relation/N,economic/J_relation/N,trade/N_relation/N,bilateral/J_relation/N,trading/N_relation/N

Add, great/J_power/N, strong/J_power/N,big/J_power/N,similar/J_power/N,real/J_power/N,high/J_power/N
Mult, great/J_power/N, complete/J_power/N,bad/J_tooth/N,complete/J_copy/N,great/J_tooth/N,perfect/J_tooth/N
Left, great/J_power/N, great/J_example/N,great/J_promise/N,great/J_person/N,great/J_britain/N,great/J_strategist/N
Right, great/J_power/N, atomic/J_power/N,satirical/J_power/N,water/N_power/N,carolina/N_power/N,authoritative/J_power/N
Baroni, great/J_power/N, economic/J_power/N,main/J_cause/N,main/J_factor/N,old/J_shoe/N,main/J_source/N
Observed, great/J_power/N, key/J_component/N,star/N_attraction/N,economic/J_power/N,biggest/J_obstacle/N,main/J_cause/N
APDT, great/J_power/N, absolute/J_power/N,psychological/J_power/N,strong/J_power/N,big/J_power/N,book/N_mix/N
Socher, great/J_power/N, great/J_life/N,great/J_care/N,great/J_relief/N,big/J_power/N,brilliant/J_power/N

Add, final/J_year/N, fourth/J_year/N,year/N_port/N,year/N_profit/N,profit/N_year/N,year/N_profit/N
Mult, final/J_year/N, dallas/N_division/N,courtroom/N_drama/N,match/N_year/N,april/N_merger/N,video/N_area/N
Left, final/J_year/N, final/J_say/N,final/J_outcome/N,final/J_impression/N,final/J_day/N,final/J_understanding/N
Right, final/J_year/N, great/J_year/N,bp/N_year/N,additional/J_year/N,higher/N_year/N,net/J_year/N
Baroni, final/J_year/N, great/J_film/N,great/J_act/N,final/J_weekend/N,early/J_month/N,early/J_year/N
Observed, final/J_year/N, year/N_cost/N,week/N_meeting/N,year/N_export/N,great/J_film/N,ice/N_age/N
APDT, final/J_year/N, following/J_year/N,public/J_week/N,calendar/N_day/N,comparable/J_month/N,16th/J_year/N
Socher, final/J_year/N, final/J_day/N,final/J_week/N,new/J_year/N,final/J_quarter/N,fourth/J_year/N

Add, good/J_portion/N, good/J_judgment/N,good/J_layer/N,good/J_entertainment/N,good/J_episode/N,good/J_novel/N
Mult, good/J_portion/N, independence/N_day/N,good/J_sale/N,major/J_portion/N,long/J_series/N,distribution/N_distribution/N
Left, good/J_portion/N, good/J_case/N,good/J_point/N,good/J_day/N,good/J_ending/N,good/J_aswell/N
Right, good/J_portion/N, later/J_portion/N,substantial/J_portion/N,cash/N_portion/N,dlr/N_portion/N,significant/J_portion/N
Baroni, good/J_portion/N, good/J_connelly/N,good/J_gag/N,good/J_jazz/N,good/J_ending/N,good/J_buy/N
Observed, good/J_portion/N, high/J_performance/N,greater/J_efficiency/N,commercial/J_application/N,right/J_kind/N,soccer/N_game/N
APDT, good/J_portion/N, huge/J_fight/N,good/J_sale/N,terrible/J_balance/N,pre-tax/J_benefit/N,second/J_voice/N
Socher, good/J_portion/N, significant/J_portion/N,high/J_portion/N,substantial/J_portion/N,considerable/J_portion/N,generous/J_portion/N

Add, high/J_esteem/N, high/J_adventure/N,high/J_comedy/N,high/J_teens/N,high/J_thriller/N,high/J_shower/N
Mult, high/J_esteem/N, high/J_movie/N,low/J_movie/N,cartoon/N_trick/N,high/J_output/N,low/J_budget/N
Left, high/J_esteem/N, high/J_corp/N,high/J_ct/N,high/J_margin/N,high/J_risk/N,high/J_cruise/N
Right, high/J_esteem/N, self/N_esteem/N,roman/J_epic/N,sci-fi/J_epic/N,hollywood/N_epic/N,page/N_epic/N
Baroni, high/J_esteem/N, recent/J_time/N,black/J_suit/N,final/J_analysis/N,domestic/J_affair/N,recent/J_year/N
Observed, high/J_esteem/N, southern/J_desert/N,central/J_london/N,colorful/J_costume/N,china/N_sea/N,recent/J_time/N
APDT, high/J_esteem/N, critical/J_revulsion/N,four-star/J_film/N,spin-off/N_asset/N,twisted/J_mind/N,intense/J_love/N
Socher, high/J_esteem/N, self/N_esteem/N,high/J_horror/N,high/J_burden/N,high/J_gear/N,high/J_speed/N

Add, financial/J_control/N, current/J_control/N,legal/J_control/N,future/J_control/N,entire/J_control/N,overall/J_control/N
Mult, financial/J_control/N, financial/J_pressure/N,electronic/J_surveillance/N,non-binding/J_agreement/N,confidential/J_agreement/N,immediate/J_control/N
Left, financial/J_control/N, financial/J_viability/N,financial/J_house/N,financial/J_income/N,financial/J_obligation/N,financial/J_activity/N
Right, financial/J_control/N, nominal/J_control/N,group/N_control/N,exchange/N_control/N,csr/N_control/N,maker/N_control/N
Baroni, financial/J_control/N, various/J_type/N,economic/J_reform/N,young/J_prostitute/N,various/J_film/N,tax/N_concession/N
Observed, financial/J_control/N, recent/J_stand/N,various/J_type/N,new/J_drug/N,comprehensive/J_package/N,lung/N_cancer/N
APDT, financial/J_control/N, export/N_control/N,oil/N_control/N,computer/N_control/N,import/N_control/N,monetary/J_control/N
Socher, financial/J_control/N, legal/J_control/N,banking/N_control/N,internal/J_control/N,monetary/J_control/N,foreign/J_control/N

Add, early/J_flight/N, early/J_settlement/N,early/J_afternoon/N,peaceful/J_planet/N,later/J_climb/N,early/J_century/N
Mult, early/J_flight/N, hard/J_series/N,impossible/J_series/N,typical/J_series/N,awesome/J_series/N,boring/J_series/N
Left, early/J_flight/N, early/J_day/N,early/J_dealing/N,early/J_summer/N,early/J_january/N,early/J_age/N
Right, early/J_flight/N, two-hour/J_flight/N,ill-fated/J_flight/N,rickety/J_flight/N,black/J_flight/N,old/J_flight/N
Baroni, early/J_flight/N, business/N_trip/N,different/J_aspect/N,various/J_matter/N,final/J_day/N,final/J_play/N
Observed, early/J_flight/N, bus/N_trip/N,10th/J_day/N,quarter/N_hour/N,business/N_trip/N,different/J_aspect/N
APDT, early/J_flight/N, current/J_day/N,long/J_market/N,extended/J_line/N,movie/N_day/N,final/J_project/N
Socher, early/J_flight/N, early/J_output/N,early/J_relief/N,early/J_film/N,early/J_movie/N,early/J_march/N

Add, market/N_fear/N, market/N_increase/N,market/N_stability/N,market/N_assistance/N,market/N_research/N,market/N_performance/N
Mult, market/N_fear/N, disney/N_production/N,market/N_shortfall/N,market/N_accounting/N,production/N_cost/N,market/N_leadership/N
Left, market/N_fear/N, market/N_report/N,market/N_talk/N,market/N_participant/N,market/N_geneticist/N,market/N_competitiveness/N
Right, market/N_fear/N, stephen/N_fear/N,worst/J_fear/N,trade/N_fear/N,immediate/J_fear/N,investor/N_fear/N
Baroni, market/N_fear/N, final/J_destination/N,certain/J_charm/N,local/J_theater/N,certain/J_minimum/N,higher/J_borrowing/N
Observed, market/N_fear/N, recent/J_complaint/N,red/J_planet/N,sell/N_recommendation/N,dry/J_land/N,optimistic/J_expectation/N
APDT, market/N_fear/N, inflation/N_fear/N,budget/N_increase/N,health/N_claim/N,price/N_reduction/N,small/J_increase/N
Socher, market/N_fear/N, trade/N_fear/N,investor/N_fear/N,psychological/J_fear/N,market/N_manipulation/N,war/N_fear/N

Add, international/J_banker/N, international/J_scientist/N,international/J_inc./N,international/J_supplier/N,international/J_syndicate/N,international/J_bidder/N
Mult, international/J_banker/N, second/J_device/N,international/J_trader/N,second/J_plant/N,second/J_punch/N,international/J_network/N
Left, international/J_banker/N, international/J_precedent/N,international/J_association/N,international/J_competitiveness/N,international/J_problem/N,international/J_forum/N
Right, international/J_banker/N, european/J_banker/N,western/J_banker/N,local/J_banker/N,long-term/J_banker/N,investment/N_banker/N
Baroni, international/J_banker/N, oil/N_unit/N,military/J_obligation/N,high/J_yen/N,international/J_bank/N,group/N_plc/N
Observed, international/J_banker/N, tycoon/N_trump/N,fellow/J_american/N,oil/N_unit/N,community/N_bank/N,evil/J_deed/N
APDT, international/J_banker/N, american/J_banker/N,mexican/J_mother/N,australian/J_nurse/N,commercial/J_banker/N,business/N_owner/N
Socher, international/J_banker/N, commercial/J_banker/N,private/J_banker/N,local/J_banker/N,foreign/J_banker/N,international/N_corporation/N

Add, world/N_centre/N, world/N_recession/N,world/N_strategy/N,world/N_management/N,world/N_supply/N,film/N_world/N
Mult, world/N_centre/N, garbage/N_world/N,world/N_attraction/N,world/N_star/N,world/N_board/N,world/N_money/N
Left, world/N_centre/N, world/N_leader/N,world/N_meeting/N,world/N_policy/N,world/N_stake/N,world/N_inc./N
Right, world/N_centre/N, overseas/J_centre/N,trade/N_centre/N,investment/N_centre/N,home/N_centre/N,financial/J_centre/N
Baroni, world/N_centre/N, world/N_center/N,world/N_debtor/N,possible/J_sanction/N,oil/N_facility/N,oil/N_installation/N
Observed, world/N_centre/N, world/N_center/N,free/J_talk/N,free/J_agreement/N,danish/J_crown/N,computer/N_subsidiary/N
APDT, world/N_centre/N, power/N_house/N,city/N_facility/N,famous/J_hall/N,trust/N_estate/N,art/N_house/N
Socher, world/N_centre/N, world/N_center/N,major/J_centre/N,world/N_club/N,world/N_tour/N,important/J_centre/N

Add, special/J_charge/N, future/J_charge/N,certain/J_charge/N,criminal/J_charge/N,gross/J_charge/N,additional/J_charge/N
Mult, special/J_charge/N, special/J_squad/N,preliminary/J_proposal/N,non-stop/J_death/N,current/J_collapse/N,historical/J_theory/N
Left, special/J_charge/N, special/J_worry/N,special/J_protection/N,special/J_murphy/N,special/J_write-off/N,special/J_stock/N
Right, special/J_charge/N, u.s./N_charge/N,after-tax/J_charge/N,sec/N_charge/N,pretax/J_charge/N,federal/N_charge/N
Baroni, special/J_charge/N, local/J_operation/N,political/J_action/N,new/J_judge/N,business/N_equipment/N,american/J_city/N
Observed, special/J_charge/N, policy/N_decision/N,illegal/J_fund/N,day-to-day/J_operation/N,boxing/N_career/N,local/J_operation/N
APDT, special/J_charge/N, cash/N_market/N,insurance/N_charge/N,black/J_market/N,hit/N_ratio/N,card/N_market/N
Socher, special/J_charge/N, formal/J_charge/N,federal/N_charge/N,financial/J_charge/N,financial/N_charge/N,credit/N_charge/N

Add, current/J_environment/N, current/J_culture/N,financial/J_health/N,current/J_earnings/N,current/J_transaction/N,current/J_expansion/N
Mult, current/J_environment/N, past/J_scene/N,past/J_period/N,past/J_event/N,past/J_game/N,past/J_competition/N
Left, current/J_environment/N, current/J_barrel/N,current/J_crop/N,current/J_qtr/N,current/J_condition/N,current/J_share/N
Right, current/J_environment/N, development/N_environment/N,new/J_environment/N,quiet/J_environment/N,investment/N_environment/N,bad/J_environment/N
Baroni, current/J_environment/N, best/J_condition/N,recent/J_session/N,total/J_security/N,market/N_segment/N,french/J_history/N
Observed, current/J_environment/N, three-piece/J_suit/N,constant/J_fear/N,miserable/J_condition/N,current/J_climate/N,monthly/J_report/N
APDT, current/J_environment/N, friendly/J_environment/N,volatile/J_coverage/N,economic/J_climate/N,latest/J_project/N,value/N_accounting/N
Socher, current/J_environment/N, current/J_problem/N,current/J_climate/N,current/J_culture/N,current/J_imbalance/N,current/J_arrangement/N

Add, public/J_sector/N, private/J_sector/N,industrial/J_sector/N,different/J_sector/N,traditional/J_sector/N,individual/J_sector/N
Mult, public/J_sector/N, hollywood/N_mind/N,form/N_letter/N,hanover/N_germany/N,june/N_statement/N,public/J_february/N
Left, public/J_sector/N, public/J_document/N,public/J_dlr/N,public/J_comment/N,public/J_march/N,public/J_price/N
Right, public/J_sector/N, technology/N_sector/N,diagnostic/N_sector/N,british/J_sector/N,computer/N_sector/N,individual/J_sector/N
Baroni, public/J_sector/N, big/J_city/N,various/J_form/N,industrial/J_sector/N,different/J_region/N,small/J_town/N
Observed, public/J_sector/N, service/N_sector/N,construction/N_industry/N,western/N_europe/N,west/N_virginia/N,abu/N_dhabi/N
APDT, public/J_sector/N, financial/J_sector/N,movie/N_category/N,gas/N_sector/N,construction/N_sector/N,traditional/J_sector/N
Socher, public/J_sector/N, industrial/J_sector/N,consumer/N_sector/N,economic/J_sector/N,private/J_sector/N,financial/J_sector/N

Add, young/J_artist/N, young/J_bird/N,young/J_actor/N,young/J_accomplice/N,young/J_viewer/N,young/J_whale/N
Mult, young/J_artist/N, young/J_psychologist/N,young/J_father/N,sick/J_factor/N,perfect/J_man/N,young/J_troupe/N
Left, young/J_artist/N, young/J_counterpart/N,young/J_scientist/N,young/J_celebrity/N,young/J_jack/N,young/J_skywalker/N
Right, young/J_artist/N, wacky/J_artist/N,inner/J_artist/N,sensitive/J_artist/N,post-post-feminist/J_artist/N,unknown/J_artist/N
Baroni, young/J_artist/N, best/J_performer/N,business/N_combination/N,higher/J_learning/N,small/J_thing/N,japanese/J_year/N
Observed, young/J_artist/N, science/N_fiction/N,raw/J_sugar/N,best/J_performer/N,italian/J_man/N,preferred/J_candidate/N
APDT, young/J_artist/N, teenage/J_artist/N,feminist/J_artist/N,popular/J_artist/N,american/J_artist/N,young/J_filmmaker/N
Socher, young/J_artist/N, young/J_actor/N,female/N_artist/N,young/J_officer/N,talented/J_artist/N,young/J_expert/N

Add, political/J_analyst/N, stiff/J_witt/N,economic/J_analyst/N,armed/J_dealer/N,political/J_source/N,major/J_broker/N
Mult, political/J_analyst/N, social/J_irony/N,historical/J_involvement/N,economic/J_analyst/N,political/J_furore/N,political/J_reason/N
Left, political/J_analyst/N, political/J_crossroads/N,political/J_wag/N,political/J_leadership/N,political/J_situatin/N,political/J_favor/N
Right, political/J_analyst/N, financial/N_analyst/N,economic/J_analyst/N,effective/J_analyst/N,cium/N_analyst/N,trade/N_analyst/N
Baroni, political/J_analyst/N, trade/N_source/N,political/J_source/N,bank/N_spokesman/N,market/N_source/N,government/N_spokesman/N
Observed, political/J_analyst/N, defense/N_analyst/N,western/J_diplomat/N,airline/N_official/N,industry/N_analyst/N,chairman/N_thompson/N
APDT, political/J_analyst/N, oil/N_analyst/N,western/J_banker/N,policy/N_analyst/N,financial/J_analyst/N,international/J_dealer/N
Socher, political/J_analyst/N, security/N_analyst/N,economic/J_analyst/N,political/J_strategist/N,media/N_analyst/N,financial/N_analyst/N

Add, new/J_offering/N, new/J_accounting/N,new/J_liquidity/N,new/J_certificate/N,new/J_borrowing/N,new/J_flat/N
Mult, new/J_offering/N, new/J_oil/N,cuban/J_money/N,new/J_job/N,iraqi/J_export/N,federal/J_incentive/N
Left, new/J_offering/N, new/J_address/N,new/J_link/N,new/J_certificate/N,new/J_baby/N,new/J_cop/N
Right, new/J_offering/N, february/N_offering/N,debenture/N_offering/N,latest/J_offering/N,animated/J_offering/N,senator/N_offering/N
Baroni, new/J_offering/N, new/J_kid/N,new/J_certificate/N,new/J_acquisition/N,new/J_swap/N,new/J_spree/N
Observed, new/J_offering/N, higher/J_value/N,fund/N_investment/N,worldwide/J_revenue/N,stock/N_dividend/N,ritual/N_killing/N
APDT, new/J_offering/N, costly/J_investment/N,common/J_offering/N,new/J_film/N,maintenance/N_operation/N,quality/N_offering/N
Socher, new/J_offering/N, new/J_offer/N,new/J_standard/N,new/J_focus/N,new/J_equivalent/N,new/N_issue/N

Add, financial/J_deficit/N, current/J_deficit/N,financial/J_datum/N,financial/J_commitment/N,fiscal/J_deficit/N,financial/J_income/N
Mult, financial/J_deficit/N, persistent/J_deficit/N,bilateral/J_deficit/N,huge/J_deficit/N,massive/J_deficit/N,continued/J_imbalance/N
Left, financial/J_deficit/N, financial/J_viability/N,financial/J_house/N,financial/J_income/N,financial/J_obligation/N,financial/J_activity/N
Right, financial/J_deficit/N, eventual/N_deficit/N,cash/N_deficit/N,february/N_deficit/N,sector/N_deficit/N,bangladesh/N_deficit/N
Baroni, financial/J_deficit/N, small/J_pool/N,financial/J_structure/N,financial/J_difficulty/N,group/N_inc./N,large/J_number/N
Observed, financial/J_deficit/N, ongoing/J_trial/N,cash/N_dividend/N,annual/J_expense/N,economic/J_environment/N,parliamentary/J_business/N
APDT, financial/J_deficit/N, cable/N_operation/N,cash/N_deficit/N,aircraft/N_practice/N,growth/N_business/N,price/N_support/N
Socher, financial/J_deficit/N, energy/N_deficit/N,financial/J_surplus/N,market/N_deficit/N,payment/N_deficit/N,trade/N_deficit/N

Add, tax/N_law/N, compensation/N_law/N,law/N_film/N,law/N_suit/N,texas/N_law/N,law/N_movie/N
Mult, tax/N_law/N, tax/N_rule/N,tax/N_regime/N,tax/N_agreement/N,deal/N_cabinet/N,england/N_pressure/N
Left, tax/N_law/N, tax/N_purpose/N,tax/N_recovery/N,tax/N_reduction/N,tax/N_code/N,tax/N_campaign/N
Right, tax/N_law/N, character-economy/J_law/N,island/N_law/N,new/N_law/N,local/J_law/N,sodbuster/N_law/N
Baroni, tax/N_law/N, military/J_basis/N,trade/N_figure/N,military/J_personnel/N,trading/N_prospects/N,special/J_troops/N
Observed, tax/N_law/N, military/J_basis/N,ground/N_troops/N,employment/N_statistics/N,employment/N_datum/N,trade/N_figure/N
APDT, tax/N_law/N, insurance/N_law/N,commercial/J_law/N,budget/N_law/N,new/J_law/N,regional/J_law/N
Socher, tax/N_law/N, tax/N_legislation/N,credit/N_law/N,investment/N_law/N,share/N_law/N,tax/N_decision/N
'''