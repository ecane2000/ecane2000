import spacy
nlp = spacy.blank("yo") 
doc = nlp("Në të dalë të dimrit, kur dërgata e sulltanit turk u largua, ne e kuptuam se lufta ishte e pashmangshme. Ata i bënë të gjitha llojet e trysnive që ne të pranonim të ktheheshim në vartës ose vasalë, siç i thonë latinët. Pas lajkave dhe premtimeve se do të na bënin bashkëqeverisës në perandorinë e tyre të paskaj, na fajësuan si rimohues, që iu jemi shitur frëngjve, domethënë Evropës. Së fundi, ashtu siç pritej, erdhën kërcënimet. Ju u besoni shumë mureve të kështjellave tuaja, thanë ata, por edhe në qoftë se ato janë vërtet ashtu siç kujtoni ju, ne do t’i rrethojme me unazën e hekurt të etjes dhe të urisë. Ne do të bëjmë që, sa herë që të vijë koha e korrjeve dhe e ditëve të lëmit, ju të shikoni qiellin sikur të ishte fushë e mbjellë dhe hëna t’ju duket si drapër. Pastaj ikën. Gjatë gjithë muajit mars lajmësit e tyre, duke rendur si era, u çuan letra vasalëve ballkanas, që, ose të na ndërronin mendjen, ose të na kthenin krahët. Dhe ata, ashtu siç pritej, bënë të dytën. Të mbetur vetëm, ne e dinim se herët ose vonë ata do të vinin. Kishim pritur kaq herë ushtri të ndryshme, por pritja e ushtrisë më të madhe të botës ishte tjetër gjë. Në kryet tona diçka ndodhte pa pushim, ndaj merrej me mend se ç’ndodhte në kryet e princit tonë Gjergj Kastriotit. Gjithë fortesat, të tokës e të anëdetit, morën urdhrat e tij për riparimin e pirgjeve e sidomos grumbullimin e armëve dhe të tëmotjeve. Ende nuk dihej se nga ç’kah do të vinin dhe, veç në fillim të qershorit, mbërriti lajmi se ishin nisur nëpër rrugën e moçme Egnatia, ç’ka do të thoshte drejt nesh. Një javë më pas, meqë kështjellës sonë i ra fati të pengonte e para dyndjen e tyre, nga kisha e madhe e Shkodrës sollën korën e Zonjës Shën Mëri, atë që dyqind vjet më parë i kishte ndihur mbrojtësit e Durrësit kundër normandëve. E përhiruam të gjithë Zonjën tonë Hyjlindëse dhe shpirtin e ndjemë më të qetë e më të fortë. Ushtria e tyre lëvizte ngadalë. Në mesin e muajit qershor kapërceu kufirin. Dhe ditë më pas Gjergj Kastrioti, i shoqëruar nga konti Muzakë, erdhi të mbikqyrte për herë të fundit fortesën e të lamtumirohej me ne. Pasi dha porositë e fundit, në mbasditen e së dielës doli nga kështjella bashkë me shpurën e tij dhe me gratë e fëmijët e oficerëve, që do të strehoheshin në male. I përcollëm një copë udhë në heshtje. Pastaj si u falëm e u përfalëm u kthyem në kështjellë. Që nga pirgjet i ndoqëm me sy gjersa dolën te Pllaja e Kryqit, në të Përpjetën e Keqe dhe së fundi te Gryka e Erës. I mbyllëm portat e rënda dhe kështjella na u duk e shurdhër pa zërat e camërdhokëve. Pastaj mbyllëm portat e dyta dhe u heshtuam edhe më. Në mesin e 13 qershorit, ndaj të gdhirë ra këmbana e kishëzës. Roja e pirgut lindor kishte pikasur në largësi një si mjegull të verdhë. Ishte pluhuri i tyre")
print([(w.text) for w in doc])
