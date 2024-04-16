import os
import sys
os.environ['NCCL_DEBUG']='WARN'

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import (get_tokenizer)
import json
short_prompt = "who found Microsoft"
long_prompt = """
read this: THE FIRST PORTAL through which entertainment slithered into life and then conquered it was journalism. As the sociologist Robert E. Park described the process in 1927, “[T]he reason we have newspapers at all in the modern sense of the term, is because about one hundred years ago, in 1835 to be exact, a few newspaper publishers in New York City and in London discovered (1) that most human beings, if they could read at all, found it easier to read news than editorial opinion and (2) that the common man would rather be entertained than edified.” Though news as entertainment was not entirely unheard of before the 1830s—one paper named Hawk & Buzzard subsisted in New York from 1826 through 1833 largely on gossip-Park was essentially correct. Prior to the 1830s most American newspapers weren’t newspapers at all. They were party broadsheets largely devoted to advertisements and partisan editorializing so rabid that Tocqueville attacked the American journalist as an uneducated vulgarian who makes “an open and coarse appeal to the passions of his readers; he abandons principles to assail the characters of individuals, to track them into private life and disclose all their weaknesses and vices.” When Benjamin Day founded the New York Sun in 1833, in the flush of Jacksonian egalitarianism, he was breaking that journalistic mold. Before the Sun, the target audience for most papers was the wealthy and the professional classes. Day explicitly appealed to “mechanics and the masses generally.” Before the Sun, most papers cost six cents. The Sun cost a penny; hence the name “penny press” that attached to it and its imitators. Before the Sun, a typical daily newspaper in New York City could expect to sell roughly 1,200 copies, with the total circulation of all eleven daily papers in the city in 1833 reaching only 26,500. After the appearance of the penny press, readership skyrocketed. In June 1835, by one report, the combined circulation of the penny papers alone was 44,000. But the real difference between these new papers and the traditional press-and the real reason for their swelling circulation-was their content. If the six-penny papers were primarily opinion sheets, the penny papers were news organs, the very first daily news organs in the country. In the penny papers one could at long last read about life in the city, the nation, even the world, and discover not what an editor thought but what people had done, or at least what they were purported to have done. Nor was it only a matter of news; it was also a matter of purview. According to the press historian Michael Schudson, the penny papers were the first to acknowledge the importance of everyday life and the first to promote the “human interest story,” which would soon become a journalistic staple. Still, the success of the penny press raised an inevitable question: Why did news rather than opinion appeal to the mass reader? One could certainly attribute the allure of the news to the need among atomized citizens in burgeoning urban areas for some sense of common experience such as news provided. Or one could attribute it to the ability of the news to reinforce the suspicion on the part of many citizens that depravity lurked just beneath the city’s surface, a suspicion that undermined the moral authority of the local elites. Still another factor may have been an intensifying sense in people that they had to know what was happening because as technology shrank the community and the nation, events that once seemed distant might now impact on their lives. No doubt all these factors, as well as others, played some role. But the single most important attraction of the penny press may have been the most obvious one—namely, that for a constituency being conditioned by trashy crime pamphlets, gory novels and overwrought melodramas, news was simply the most exciting, most entertaining content a paper could offer, especially when it was skewed, as it invariably was in the penny press, to the most sensational stories. In fact, one might even say that the masters of the penny press invented the concept of news because it was the best way to sell their papers in an entertainment environment, and it was certainly no small matter that while the six-penny papers were sold primarily through subscription, the penny papers were hawked on the streets, meaning that the content had to be interesting enough to entice readers into buying a paper. The publishers of the penny press didn’t necessarily protest the idea that they were in the entertainment rather than the information business. From its inception the penny press began specializing in crime, with an emphasis on murder, to distinguish it from what the New York Herald called the “dull business air of the large morning papers.” In its first two weeks of publication in May 1835, the Herald itself featured three suicides, three murders, the death of five persons in a fire, a man accidentally blowing off his head, an execution in France by guillotine and a riot in Philadelphia. But the Herald’s real breakthrough as an entertainment medium came a year later, when it pounced on the case of a murdered nineteen-year-old prostitute named Helen Jewett to build its circulation. The Jewett case had all the hallmarks of the stories that would dominate the tabloid press a hundred years later, and the tabloid television news shows fifty years after that, On the night of her death, April 9, 1836, which also happened to be her birthday, the victim had entertained a prosperous young clerk named Richard P. Robinson, who had recently become affianced to a woman of good pedigree and who had visited Jewett, said her madam, Rosina Townsend, to retrieve some items he had given her. Townsend had seen them together in Jewett’s room at eleven o’clock that evening. When Townsend found the young woman’s battered and bloody body in a smoldering bed early the next morning, Robinson was the obvious suspect. On the evidence, Robinson certainly seemed guilty. A cloak found in the yard behind the brothel was traced to him. The ax that had struck the killing blows was identified as one the defendant had used to chop wood. Whitewash from the back fence was discovered on his trousers. A pharmacist testified that Robinson had purchased arsenic a week earlier. The accused’s roommate confessed that Robinson had been out late the night of the murder. Nevertheless, at his trial that June it took the jury only ten minutes to deliver a verdict of not guilty. The significance of the Jewett case, however, had little to do with jurisprudence or justice. Its significance lay rather in what it mainlined into the bloodstream of the American press: the incalculable entertainment value of a lurid or prurient tale. Not only did the story demonstrate how the press might contour news to the hoariest conventions of melodrama, it showed as well just how quickly the press came to appreciate the appeal of these conventions in a realistic context and how quickly it learned to exploit them, thus setting the terms for the American press forevermore. The keenest of these new press barons was James Gordon Bennett, Scottish-born steward of the Herald who would later gain notoriety for his scurrilous methods and crude behavior (He once urinated in his fiancée’s piano during a soiree.) Bennett realized he was on to something good with the Jewett story and did everything he could to milk it. There was a long, scandalous profile of the victim, as overwrought as anything in the sentimental novels of Susan Warner, portraying Jewett as a poor good girl seduced by a rogue and then abandoned, a ruination that set the course for her wayward life. There was what purported to be a firsthand description by Bennett of the victim’s room while the body still lay in it. There was an interview conducted by Bennett with Rosina Townsend (though Townsend denied having given it). And there was story after story assessing Robinson’s guilt or innocence, first listing this way and then that before finally settling on Townsend as the real murderer, abetted by the police and, of all people, Bennett’s chief penny press rival, Benjamin Day of the Sun. Unsurprisingly, critics decried what the Jewett case had unleashed. Charles Dickens, after a visit to America in 1842, wrote that no matter what Americans did, “while the newspaper press of America is in, or near, its present abject state, high moral improvement in that country is hopeless.” Within a few years of the Jewett case, a coalition of clergymen, financiers, rival editors and Van Buren Democrats, all of whom had been offended by Bennett, launched what came to be called a “Moral War” against the Herald, pressuring readers, advertisers and distributors not to read, advertise in or distribute the paper—in effect, to place it in Coventry. One warrior accused Bennett of “moral leprosy.” But, as with the criticism of conventional entertainments, the issue really had less to do with morality than with cultural control. The Herald, as an engine of trashy entertainment, challenged the genteel social order and did so in a new arena outside the boundaries of traditional entertainment, which seemed to make it even more invidious. In attacking the paper, then, the genteel elites were once again trying to destroy an institution that clearly threatened their authority. And once again, they were right to be alarmed. Though he occasionally affected a concern for moral values himself, Bennett, drawing as had other entertainers on the power of Jacksonian democracy, was a born agitator. He intended, said his official biographer, to rescue his readers from “affected prudery” and “mawkish refinement,” and took pride in claiming that he had “entered the hearts of the people,” “shown them their own sentiments” and “put down their own living feelings on paper…” In the end, the elites were no more successful in suppressing the penny press than they had been or would be in suppressing other entertainments, especially since their condemnation was itself part of the appeal of the penny papers, Bennett’s Herald boasted a daily circulation of 20,000 during the Robinson trial and 51,000 by the time of the Moral War—larger than the total circulation of the papers run by what he sneeringly called the “Holy Allies.” To his critics’ everlasting regret, he knew that his readers, as he would later say, “were more ready to seek six columns of the details of a brutal murder, or the testimony of a divorce case, or the trial of a divine for improprieties of conduct, than the same amount of words poured forth by the genius of the noblest author of the times.” Even the Civil War, the nation’s bloodiest tragedy, could not escape exploitation by the sensationalist impulse. Newspaper sales soared during the peaks of conflict—Bennett sold 135,000 copies of the Herald with the attack on Fort Sumter and published three daily editions thereafter-then dropped when the action seemed to flag, and some seemed to see a correlation not between information and circulation but between entertainment and circulation. “A week passed without reports of a battle with thousands killed and wounded,” complained Colonel Charles S. Wainwright sarcastically, “is very dull, rendering the papers hardly worth reading.” Another observer worried that the public’s taste for sensation might actually prolong the war just so they could get more of it. “When it was feared and believed that General Lee might take Washington, Philadelphia, and even New York,” he wrote, “there was no panic in those cities, nothing beyond a new sensation, which I believe they enjoyed as much as the spectators of Blondid and Leotard [two daredevils] did their feats of daring and danger.” Critics generally assumed that the curse of sensationalism was the legacy bequeathed by Bennett’s Herald to the American press. “He made the newspaper powerful, but he made it odious,” was how rival editor Horace Greeley of the New York Tribune put it. What was a less obvious but arguably a far more important legacy, is that Bennett breached the walls that divided imagination from reality and separated clearly defined entertainments like the theatrical drama, the dime novel and the musicale from a kind of entertainment for which there is still no name, perhaps because people are still loath to acknowledge that it is entertainment—an entertainment in life. In short, by inventing the news, Bennett confused realms so thoroughly that no one would ever be able to resolve the confusion. Thus did he confirm Edgar Allan Poe’s prescient observation: that the penny press affected “the interests of the country at large” in ways "probably beyond all calculation". Robert is a
"""

# print(len(long_prompt))
# prompts = [long_prompt[10000]]

prompts = [long_prompt]

# prompts = [short_prompt]

model_path="/data/users/yunanzhang/hf/checkpoints/TLG4.7.3/iter_0078678_hf/"
sampling_params = SamplingParams(temperature=0)
llm = LLM(model=model_path,tokenizer='TLGv4Tokenizer',enforce_eager=True)
outputs = llm.generate(prompts,sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"result:\n {generated_text}")