import json
import os
import re
from typing import List, Dict, Optional, Tuple

# Province abbreviation mapping (comprehensive)
province_dict = {
    "沪": "上海",
    "京": "北京",
    "吉": "吉林",
    "川": "四川",
    "蜀": "四川",
    "津": "天津",
    "皖": "安徽",
    "安": "安徽",
    "鲁": "山东",
    "晋": "山西",
    "粤": "广东",
    "桂": "广西",
    "苏": "江苏",
    "赣": "江西",
    "冀": "河北",
    "豫": "河南",
    "浙": "浙江",
    "鄂": "湖北",
    "湘": "湖南",
    "黔": "贵州",
    "贵": "贵州",
    "辽": "辽宁",
    "渝": "重庆",
    "陕": "陕西",
    "秦": "陕西",
    "青": "青海",
    "黑": "黑龙江"
}

missing_provinces_dict = {
    "云": "云南",
    "蒙": "内蒙古",
    "宁": "宁夏",
    "新": "新疆",
    "海": "海南",
    "甘": "甘肃",
    "闽": "福建",
    "藏": "西藏"
}

# Merge all province mappings
ALL_PROVINCE_ABBR = {**province_dict, **missing_provinces_dict}

# Complete list of provinces with data in knowledge base
shengfen_li = [
    "上海", "北京", "吉林", "四川", "天津", "安徽", "山东", "山西",
    "广东", "广西", "江苏", "江西", "河北", "河南", "浙江", "湖北", 
    "湖南", "贵州", "辽宁", "重庆", "陕西", "青海", "黑龙江"
]

# Provinces without direct data (use adjacent provinces as fallback)
missing_provinces = ['云南', '内蒙古', '宁夏', '新疆', '海南', '甘肃', '福建', '西藏']

# Adjacent provinces mapping for fallback mechanism
adjacent_provinces = {
    '云南': ["广西"],
    '内蒙古': ['黑龙江'],
    '宁夏': ['陕西'],
    '新疆': ['陕西'],
    '海南': ['广东'],
    '甘肃': ["陕西"],
    '福建': ["浙江", "江西", "广东"],
    '西藏': ["四川"]
}

# Charge normalization dictionary
charge_dict = {
    '诈骗罪': '诈骗罪',
    '合同诈骗罪': '合同诈骗罪',
    '保险诈骗罪': '保险诈骗罪',
    '贷款诈骗罪': '贷款诈骗罪',
    '招摇撞骗罪': '招摇撞骗罪',
    '盗窃罪': '盗窃罪',
    '盗伐林木罪': '盗伐林木罪',
    '故意伤害罪': '故意伤害罪',
    '寻衅滋事罪': '寻衅滋事罪',
    '聚众斗殴罪': '聚众斗殴罪',
    '故意杀人罪': '故意杀人罪',
    '赌博罪': '赌博罪',
    '开设赌场罪': '开设赌场罪',
    '受贿罪': '受贿罪',
    '行贿罪': '行贿罪',
    '贪污罪': '贪污罪',
    '妨害公务罪': '妨害公务罪',
    '非法拘禁罪': '非法拘禁罪',
    '敲诈勒索罪': '敲诈勒索罪',
    # Drug-related charges normalization
    '贩卖毒品罪': '走私、贩卖、运输、制造毒品罪',
    '贩卖、运输毒品罪': '走私、贩卖、运输、制造毒品罪',
    '运输毒品罪': '走私、贩卖、运输、制造毒品罪',
    '制造毒品罪': '走私、贩卖、运输、制造毒品罪',
    '贩卖、制造毒品罪': '走私、贩卖、运输、制造毒品罪',
    '走私、贩卖毒品罪': '走私、贩卖、运输、制造毒品罪',
    '走私、贩卖、运输毒品罪': '走私、贩卖、运输、制造毒品罪',
    '走私、运输毒品罪': '走私、贩卖、运输、制造毒品罪',
    '走私毒品罪': '走私、贩卖、运输、制造毒品罪',
    '贩卖、运输、制造毒品罪': '走私、贩卖、运输、制造毒品罪',
    '走私、贩卖、运输、制造毒品罪': '走私、贩卖、运输、制造毒品罪',
    '制造、贩卖毒品罪': '走私、贩卖、运输、制造毒品罪',
    '非法持有毒品罪': '非法持有毒品罪',
    '窝藏毒品罪': '窝藏、转移毒品罪',
    '窝藏、转移毒品罪': '窝藏、转移毒品罪',
    '转移毒品罪': '窝藏、转移毒品罪',
    '包庇毒品犯罪分子罪': '包庇毒品犯罪分子罪'
}


class LegalKnowledgeRetriever:
    """
    Implements rule-based legal knowledge retrieval according to the mapping function 
    φ: R × C → L, where jurisdiction R and criminal charges C serve as dual indices 
    to retrieve related legal knowledge L.
    
    Follows the hierarchical fallback strategy described in the research paper:
    1. Accurate Jurisdiction Identification
    2. Comprehensive Coverage Through Fallback Mechanism
    3. Systematic Charge-based Retrieval
    """
    
    def __init__(self, kb_path: str = None, general_knowledge_path: str = None):
        """Initialize the retriever with the legal knowledge base"""
        if kb_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            kb_path = os.path.join(project_root, "LegalKnowBase", "legal_knowledge_base.json")
        if general_knowledge_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            general_knowledge_path = os.path.join(
                project_root, "LegalKnowBase", "general_knowledge_other_charges.txt"
            )
        self.kb_path = kb_path
        self.general_knowledge_path = general_knowledge_path
        self.knowledge_base = self._load_knowledge_base()
        self.general_knowledge = self._load_general_knowledge()
        
    def _load_knowledge_base(self) -> Dict:
        """Load the legal knowledge base from JSON file"""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Legal knowledge base not found at: {self.kb_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in knowledge base: {self.kb_path}")

    def _load_general_knowledge(self) -> str:
        """Load general sentencing methods and common sentencing circumstances."""
        if not self.general_knowledge_path or not os.path.exists(self.general_knowledge_path):
            return ""
        with open(self.general_knowledge_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    def identify_jurisdiction(self, case_facts: str) -> Optional[str]:
        """
        Step 1: Accurate Jurisdiction Identification
        
        Develops a comprehensive province mapping dictionary that standardizes various 
        forms of geographical references (standard names, abbreviations, and historical 
        variations) in case facts. The system identifies the first provincial reference 
        as the primary jurisdiction.
        
        Args:
            case_facts: The factual description of the case
            
        Returns:
            Standardized province name or None if no jurisdiction found
        """
        # Strategy 1: Check for full province names in order of appearance
        for province in shengfen_li:
            if province in case_facts:
                return province
        
        # Strategy 2: Check for province abbreviations (especially in first 80 chars)
        # Prioritize early mentions as they often indicate jurisdiction
        early_text = case_facts[:200]  # Check first 200 characters for abbreviations
        for abbr, province in ALL_PROVINCE_ABBR.items():
            if abbr in early_text:
                return province
        
        # Strategy 3: Check entire text for abbreviations if not found early
        for abbr, province in ALL_PROVINCE_ABBR.items():
            if abbr in case_facts:
                return province
        
        return None
    
    def apply_fallback_jurisdiction(self, identified_province: Optional[str]) -> List[str]:
        """
        Step 2: Comprehensive Coverage Through Fallback Mechanism
        
        Implements hierarchical fallback strategy:
        (1) If specific province has no guidelines, reference adjacent jurisdictions
        (2) Always include national guidelines as baseline
        
        Args:
            identified_province: The province identified from case facts
            
        Returns:
            List of jurisdictions to query (in priority order)
        """
        jurisdictions_to_query = []
        
        # Priority 1: Identified province (if available and has data)
        if identified_province:
            if identified_province in shengfen_li:
                jurisdictions_to_query.append(identified_province)
            elif identified_province in missing_provinces:
                # Use adjacent provinces as fallback
                adjacent = adjacent_provinces.get(identified_province, [])
                jurisdictions_to_query.extend(adjacent)
        
        # Priority 2: National guidelines (always included as baseline)
        jurisdictions_to_query.append("全国")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_jurisdictions = []
        for j in jurisdictions_to_query:
            if j not in seen:
                seen.add(j)
                unique_jurisdictions.append(j)
        
        return unique_jurisdictions
    
    def normalize_charges(self, raw_charges: List[str]) -> List[str]:
        """
        Normalize charge names to match knowledge base keys
        
        Args:
            raw_charges: List of charge names from case data
            
        Returns:
            List of normalized charge names
        """
        normalized = []
        for charge in raw_charges:
            # Use charge_dict for normalization, keep original if not found
            normalized_charge = charge_dict.get(charge, charge)
            normalized.append(normalized_charge)
        
        # Return unique charges while preserving order
        seen = set()
        unique_charges = []
        for c in normalized:
            if c not in seen:
                seen.add(c)
                unique_charges.append(c)
        
        return unique_charges
    
    def retrieve_knowledge_for_charge(self, jurisdiction: str, charge: str) -> Optional[Dict]:
        """
        Retrieve legal knowledge for a specific jurisdiction-charge pair
        
        Args:
            jurisdiction: Province name or "全国"
            charge: Normalized charge name
            
        Returns:
            Dictionary containing legal knowledge or None if not found
        """
        if jurisdiction == "全国":
            # Query national level
            return self.knowledge_base.get("全国", {}).get(charge)
        else:
            # Query provincial level
            province_key = self._format_province_key(jurisdiction)
            prov_data = self.knowledge_base.get("provinces", {}).get(province_key)
            
            if prov_data and "charges" in prov_data:
                return prov_data["charges"].get(charge)
        
        return None
    
    def _format_province_key(self, province_name: str) -> str:
        """
        Format province name to match knowledge base keys
        
        Args:
            province_name: Standard province name
            
        Returns:
            Formatted province key (e.g., "上海市", "广东省", "广西壮族自治区")
        """
        if province_name in ["上海", "北京", "天津", "重庆"]:
            return f"{province_name}市"
        elif province_name == "广西":
            return "广西壮族自治区"
        elif province_name == "内蒙古":
            return "内蒙古自治区"
        elif province_name == "宁夏":
            return "宁夏回族自治区"
        elif province_name == "新疆":
            return "新疆维吾尔自治区"
        elif province_name == "西藏":
            return "西藏自治区"
        else:
            return f"{province_name}省"
    
    def retrieve_legal_knowledge(self, case: Dict) -> Dict:
        """
        Main retrieval function implementing the complete φ: R × C → L mapping
        
        Based on the legal knowledge base, implements rule-based legal knowledge 
        retrieval according to the mapping function φ: R × C → L, where jurisdiction 
        R and criminal charges C serve as dual indices to retrieve related legal knowledge L.
        
        Args:
            case: Dictionary containing case information with keys:
                - 'fact': Case factual description
                - 'criminals_info': Defendant information with accusations
                
        Returns:
            Dictionary containing retrieved legal knowledge organized by:
                - jurisdiction: Identified jurisdiction(s)
                - charges: List of normalized charges
                - knowledge: Retrieved legal texts organized by jurisdiction and charge
                - retrieval_strategy: Description of fallback strategy used
        """
        # Extract case information
        case_facts = case.get('fact', '')
        
        # Step 1: Identify jurisdiction
        identified_province = self.identify_jurisdiction(case_facts)
        
        # Step 2: Apply fallback mechanism to get jurisdiction list
        jurisdictions = self.apply_fallback_jurisdiction(identified_province)
        
        # Step 3: Extract and normalize charges
        raw_charges = self._extract_charges_from_case(case)
        normalized_charges = self.normalize_charges(raw_charges)
        
        # Step 4: Systematic charge-based retrieval for multiple charges
        retrieved_knowledge = {}
        retrieval_log = []
        
        for jurisdiction in jurisdictions:
            retrieved_knowledge[jurisdiction] = {}
            
            for charge in normalized_charges:
                knowledge = self.retrieve_knowledge_for_charge(jurisdiction, charge)
                
                if knowledge:
                    retrieved_knowledge[jurisdiction][charge] = knowledge
                    retrieval_log.append({
                        'jurisdiction': jurisdiction,
                        'charge': charge,
                        'status': 'found',
                        'source': knowledge.get('source', 'Unknown')
                    })
                else:
                    retrieval_log.append({
                        'jurisdiction': jurisdiction,
                        'charge': charge,
                        'status': 'not_found'
                    })
        
        # Compile retrieval result
        result = {
            'jurisdiction': {
                'identified': identified_province,
                'queried': jurisdictions,
                'fallback_applied': identified_province in missing_provinces if identified_province else False
            },
            'charges': {
                'raw': raw_charges,
                'normalized': normalized_charges,
                'count': len(normalized_charges)
            },
            'general_knowledge': {
                'source': os.path.basename(self.general_knowledge_path)
                if self.general_knowledge_path else None,
                'content': self.general_knowledge
            },
            'knowledge': retrieved_knowledge,
            'retrieval_strategy': self._describe_retrieval_strategy(identified_province, jurisdictions),
            'retrieval_log': retrieval_log,
            'statistics': {
                'total_queries': len(jurisdictions) * len(normalized_charges),
                'successful_retrievals': sum(
                    1 for log in retrieval_log if log['status'] == 'found'
                ),
                'failed_retrievals': sum(
                    1 for log in retrieval_log if log['status'] == 'not_found'
                )
            }
        }
        
        return result
    
    def _extract_charges_from_case(self, case: Dict) -> List[str]:
        """
        Extract charges from case structure
        
        Args:
            case: Case dictionary
            
        Returns:
            List of raw charge names
        """
        charge_list = []
        criminals_info = case.get('criminals_info', {})
        
        for defendant_name, defendant_data in criminals_info.items():
            accusations = defendant_data.get('accusations', [])
            if isinstance(accusations, list):
                charge_list.extend(accusations)
            elif isinstance(accusations, str):
                charge_list.append(accusations)
        
        return charge_list
    
    def _describe_retrieval_strategy(self, identified_province: Optional[str], 
                                    jurisdictions: List[str]) -> str:
        """
        Generate human-readable description of the retrieval strategy used
        
        Args:
            identified_province: Province identified from case facts
            jurisdictions: List of jurisdictions queried
            
        Returns:
            Description string
        """
        if not identified_province:
            return "No explicit jurisdiction found. Using national guidelines only."
        
        if identified_province in shengfen_li:
            return f"Direct match: Retrieved guidelines from {identified_province} and national level."
        
        if identified_province in missing_provinces:
            adjacent = adjacent_provinces.get(identified_province, [])
            return (f"Fallback applied: '{identified_province}' has no direct guidelines. "
                   f"Using adjacent province(s) {', '.join(adjacent)} and national guidelines.")
        
        return f"Standard retrieval from {identified_province} and national level."


# Convenience functions for backward compatibility
def get_case_province(case: Dict) -> List[str]:
    """
    Legacy function - returns list of provinces found in case facts
    Now uses the improved LegalKnowledgeRetriever
    """
    retriever = LegalKnowledgeRetriever()
    province = retriever.identify_jurisdiction(case.get('fact', ''))
    return [province] if province else []


def get_case_charge(case: Dict) -> List[str]:
    """
    Legacy function - returns list of charges from case
    """
    retriever = LegalKnowledgeRetriever()
    return retriever._extract_charges_from_case(case)


def get_case_defendant_num(case: Dict) -> int:
    """
    Returns the number of defendants in the case
    """
    return len(case.get('criminals_info', {}))


def get_case_total_unique_charge_num(case: Dict) -> int:
    """
    Returns the count of unique normalized charges
    """
    retriever = LegalKnowledgeRetriever()
    raw_charges = retriever._extract_charges_from_case(case)
    normalized = retriever.normalize_charges(raw_charges)
    return len(normalized)


def get_case_total_unique_charge_list(case: Dict) -> List[str]:
    """
    Returns list of unique normalized charges
    """
    retriever = LegalKnowledgeRetriever()
    raw_charges = retriever._extract_charges_from_case(case)
    return retriever.normalize_charges(raw_charges)


# Main retrieval function (recommended to use)
def legal_knowledge_retrieval(
    case: Dict,
    kb_path: str = None,
    general_knowledge_path: str = None
) -> Dict:
    """
    Main entry point for legal knowledge retrieval
    
    Implements the complete φ: R × C → L mapping as described in the research paper.
    
    Args:
        case: Dictionary containing case information
        kb_path: Optional path to knowledge base (uses default if not specified)
        general_knowledge_path: Optional path to general sentencing knowledge
        
    Returns:
        Dictionary containing retrieved legal knowledge
        
    Example:
        >>> case = {
        ...     'fact': '被告人在上海市盗窃...',
        ...     'criminals_info': {
        ...         '张三': {'accusations': ['盗窃罪']}
        ...     }
        ... }
        >>> result = legal_knowledge_retrieval(case)
        >>> print(result['jurisdiction']['identified'])  # '上海'
        >>> print(result['charges']['normalized'])  # ['盗窃罪']
        >>> print(result['knowledge']['上海市']['盗窃罪'])  # Legal text
    """
    if kb_path or general_knowledge_path:
        retriever = LegalKnowledgeRetriever(kb_path, general_knowledge_path)
    else:
        retriever = LegalKnowledgeRetriever()
    
    return retriever.retrieve_legal_knowledge(case)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 70)
    print("LEGAL KNOWLEDGE RETRIEVAL SYSTEM TEST")
    print("=" * 70)
    
    # Test case 1: Shanghai theft case
    test_case_1 = {
        'fact': '2023年5月，被告人在上海市浦东新区实施盗窃行为，盗窃金额人民币5000元。',
        'criminals_info': {
            '张三': {
                'accusations': ['盗窃罪']
            }
        }
    }
    
    print("\n--- Test Case 1: Shanghai Theft ---")
    result1 = legal_knowledge_retrieval(test_case_1)
    print(f"Identified Jurisdiction: {result1['jurisdiction']['identified']}")
    print(f"Queried Jurisdictions: {result1['jurisdiction']['queried']}")
    print(f"Charges: {result1['charges']['normalized']}")
    print(f"Retrieval Strategy: {result1['retrieval_strategy']}")
    print(f"Successful Retrievals: {result1['statistics']['successful_retrievals']}/{result1['statistics']['total_queries']}")
    
    # Test case 2: Yunnan drug case (requires fallback)
    test_case_2 = {
        'fact': '被告人在云南省昆明市贩卖毒品海洛因10克。',
        'criminals_info': {
            '李四': {
                'accusations': ['贩卖毒品罪']
            }
        }
    }
    
    print("\n--- Test Case 2: Yunnan Drug Trafficking (Fallback) ---")
    result2 = legal_knowledge_retrieval(test_case_2)
    print(f"Identified Jurisdiction: {result2['jurisdiction']['identified']}")
    print(f"Queried Jurisdictions: {result2['jurisdiction']['queried']}")
    print(f"Fallback Applied: {result2['jurisdiction']['fallback_applied']}")
    print(f"Charges: {result2['charges']['normalized']}")
    print(f"Retrieval Strategy: {result2['retrieval_strategy']}")
    print(f"Successful Retrievals: {result2['statistics']['successful_retrievals']}/{result2['statistics']['total_queries']}")
    
    # Test case 3: Multi-charge case
    test_case_3 = {
        'fact': '被告人在北京市朝阳区先实施诈骗，后为抗拒抓捕使用暴力致人轻伤。',
        'criminals_info': {
            '王五': {
                'accusations': ['诈骗罪', '故意伤害罪']
            }
        }
    }
    
    print("\n--- Test Case 3: Beijing Multi-Charge Case ---")
    result3 = legal_knowledge_retrieval(test_case_3)
    print(f"Identified Jurisdiction: {result3['jurisdiction']['identified']}")
    print(f"Queried Jurisdictions: {result3['jurisdiction']['queried']}")
    print(f"Charges: {result3['charges']['normalized']}")
    print(f"Charge Count: {result3['charges']['count']}")
    print(f"Retrieval Strategy: {result3['retrieval_strategy']}")
    print(f"Successful Retrievals: {result3['statistics']['successful_retrievals']}/{result3['statistics']['total_queries']}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
